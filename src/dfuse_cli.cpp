/**
 * @file dfuse_cli.cpp
 * @brief dFUSE Command Line Interface - Drop-in replacement for Fortran FUSE
 * 
 * Usage:
 *   dfuse <fileManager> <basinID> <runMode>
 * 
 * Arguments:
 *   fileManager: Path to FUSE file manager (fm_*.txt)
 *   basinID: Basin identifier
 *   runMode: run_def (default parameters) or run_pre (preset)
 */

#include "dfuse/dfuse.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <chrono>
#include <regex>
#include <cmath>

#ifdef DFUSE_USE_NETCDF
#include <ncFile.h>
#include <ncDim.h>
#include <ncVar.h>
#include <ncVarAtt.h>
using namespace netCDF;
#endif

namespace fs = std::filesystem;
using namespace dfuse;

// ============================================================================
// FILE MANAGER PARSER
// ============================================================================

struct FileManager {
    std::string setngs_path;
    std::string input_path;
    std::string output_path;
    std::string suffix_forcing;
    std::string suffix_elev_bands;
    std::string forcing_info;
    std::string constraints;
    std::string mod_numerix;
    std::string m_decisions;
    std::string fmodel_id;
    bool q_only;
    std::string date_start_sim;
    std::string date_end_sim;
    std::string date_start_eval;
    std::string date_end_eval;
    std::string metric;
};

FileManager parse_file_manager(const std::string& path) {
    FileManager fm;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file manager: " + path);
    }
    
    std::vector<std::string*> fields = {
        &fm.setngs_path, &fm.input_path, &fm.output_path,
        &fm.suffix_forcing, &fm.suffix_elev_bands,
        &fm.forcing_info, &fm.constraints, &fm.mod_numerix, &fm.m_decisions,
        &fm.fmodel_id
    };
    
    std::string line;
    size_t field_idx = 0;
    
    while (std::getline(file, line) && field_idx < fields.size()) {
        // Skip header and comments
        if (line.empty() || line[0] == '!' || line.find("FUSE_FILEMANAGER") != std::string::npos) {
            continue;
        }
        
        // Extract quoted string
        size_t start = line.find('\'');
        size_t end = line.rfind('\'');
        if (start != std::string::npos && end != std::string::npos && end > start) {
            std::string value = line.substr(start + 1, end - start - 1);
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            *fields[field_idx] = value;
            field_idx++;
        }
    }
    
    // Parse remaining fields (q_only, dates, etc.)
    std::vector<std::string> remaining_fields;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '!') continue;
        size_t start = line.find('\'');
        size_t end = line.rfind('\'');
        if (start != std::string::npos && end != std::string::npos && end > start) {
            remaining_fields.push_back(line.substr(start + 1, end - start - 1));
        }
    }
    
    if (remaining_fields.size() >= 1) {
        fm.q_only = (remaining_fields[0] == "TRUE" || remaining_fields[0] == "true");
    }
    if (remaining_fields.size() >= 2) fm.date_start_sim = remaining_fields[1];
    if (remaining_fields.size() >= 3) fm.date_end_sim = remaining_fields[2];
    if (remaining_fields.size() >= 4) fm.date_start_eval = remaining_fields[3];
    if (remaining_fields.size() >= 5) fm.date_end_eval = remaining_fields[4];
    if (remaining_fields.size() >= 7) fm.metric = remaining_fields[6];
    
    return fm;
}

// ============================================================================
// DECISIONS PARSER
// ============================================================================

struct ModelDecisions {
    std::string rferr = "additive_e";
    std::string arch1 = "onestate_1";
    std::string arch2 = "unlimfrc_2";
    std::string qsurf = "arno_x_vic";
    std::string qperc = "perc_f2sat";
    std::string esoil = "rootweight";
    std::string qintf = "intflwnone";
    std::string q_tdh = "rout_gamma";
    std::string snowmod = "temp_index";
};

ModelDecisions parse_decisions(const std::string& path) {
    ModelDecisions dec;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open decisions file: " + path);
    }
    
    std::map<std::string, std::string*> decision_map = {
        {"RFERR", &dec.rferr},
        {"ARCH1", &dec.arch1},
        {"ARCH2", &dec.arch2},
        {"QSURF", &dec.qsurf},
        {"QPERC", &dec.qperc},
        {"ESOIL", &dec.esoil},
        {"QINTF", &dec.qintf},
        {"Q_TDH", &dec.q_tdh},
        {"SNOWM", &dec.snowmod}
    };
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '(' || line[0] == '-' || line[0] == '!') continue;
        if (line[0] == '0') break;  // End marker
        
        std::istringstream iss(line);
        std::string option, decision;
        if (iss >> option >> decision) {
            auto it = decision_map.find(decision);
            if (it != decision_map.end()) {
                // Convert to lowercase
                std::transform(option.begin(), option.end(), option.begin(), ::tolower);
                *it->second = option;
            }
        }
    }
    
    return dec;
}

ModelConfig decisions_to_config(const ModelDecisions& dec) {
    ModelConfig config;
    
    // Upper layer
    if (dec.arch1 == "onestate_1") config.upper_arch = UpperLayerArch::SINGLE_STATE;
    else if (dec.arch1 == "tension1_1") config.upper_arch = UpperLayerArch::TENSION_FREE;
    else if (dec.arch1 == "tension2_1") config.upper_arch = UpperLayerArch::TENSION2_FREE;
    
    // Lower layer
    if (dec.arch2 == "fixedsiz_2") config.lower_arch = LowerLayerArch::SINGLE_NOEVAP;
    else if (dec.arch2 == "unlimfrc_2" || dec.arch2 == "unlimpow_2") config.lower_arch = LowerLayerArch::SINGLE_EVAP;
    else if (dec.arch2 == "tens2pll_2") config.lower_arch = LowerLayerArch::TENSION_2RESERV;
    
    // Baseflow
    if (dec.arch2 == "unlimfrc_2") config.baseflow = BaseflowType::LINEAR;
    else if (dec.arch2 == "unlimpow_2") config.baseflow = BaseflowType::NONLINEAR;
    else if (dec.arch2 == "tens2pll_2") config.baseflow = BaseflowType::PARALLEL_LINEAR;
    else if (dec.arch2 == "topmdexp_2") config.baseflow = BaseflowType::TOPMODEL;
    
    // Surface runoff
    if (dec.qsurf == "prms_varnt") config.surface_runoff = SurfaceRunoffType::UZ_LINEAR;
    else if (dec.qsurf == "arno_x_vic") config.surface_runoff = SurfaceRunoffType::UZ_PARETO;
    else if (dec.qsurf == "tmdl_param") config.surface_runoff = SurfaceRunoffType::LZ_GAMMA;
    
    // Percolation
    if (dec.qperc == "perc_f2sat") config.percolation = PercolationType::TOTAL_STORAGE;
    else if (dec.qperc == "perc_w2sat") config.percolation = PercolationType::FREE_STORAGE;
    else if (dec.qperc == "perc_lower") config.percolation = PercolationType::LOWER_DEMAND;
    
    // Evaporation
    if (dec.esoil == "sequential") config.evaporation = EvaporationType::SEQUENTIAL;
    else if (dec.esoil == "rootweight") config.evaporation = EvaporationType::ROOT_WEIGHT;
    
    // Interflow
    if (dec.qintf == "intflwnone") config.interflow = InterflowType::NONE;
    else if (dec.qintf == "intflwsome") config.interflow = InterflowType::LINEAR;
    
    // Snow
    config.enable_snow = (dec.snowmod != "no_snowmod");
    
    return config;
}

// ============================================================================
// PARAMETER CONSTRAINTS PARSER
// ============================================================================

struct FortranParams {
    Real MAXWATR_1 = 100.0;
    Real MAXWATR_2 = 1000.0;
    Real FRACTEN = 0.5;
    Real FRCHZNE = 0.5;
    Real FPRIMQB = 0.5;
    Real RTFRAC1 = 0.75;
    Real PERCRTE = 100.0;
    Real PERCEXP = 5.0;
    Real SACPMLT = 10.0;
    Real SACPEXP = 5.0;
    Real PERCFRAC = 0.5;
    Real FRACLOWZ = 0.5;
    Real IFLWRTE = 500.0;
    Real BASERTE = 50.0;
    Real QB_POWR = 5.0;
    Real QB_PRMS = 0.01;
    Real QBRATE_2A = 0.025;
    Real QBRATE_2B = 0.01;
    Real SAREAMAX = 0.25;
    Real AXV_BEXP = 0.3;
    Real LOGLAMB = 7.5;
    Real TISHAPE = 3.0;
    Real TIMEDELAY = 0.9;
    Real MBASE = 1.0;
    Real MFMAX = 4.2;
    Real MFMIN = 2.4;
    Real PXTEMP = 1.0;
};

FortranParams parse_constraints(const std::string& path) {
    FortranParams params;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open constraints file: " + path);
    }
    
    std::map<std::string, Real*> param_map = {
        {"MAXWATR_1", &params.MAXWATR_1},
        {"MAXWATR_2", &params.MAXWATR_2},
        {"FRACTEN", &params.FRACTEN},
        {"FRCHZNE", &params.FRCHZNE},
        {"FPRIMQB", &params.FPRIMQB},
        {"RTFRAC1", &params.RTFRAC1},
        {"PERCRTE", &params.PERCRTE},
        {"PERCEXP", &params.PERCEXP},
        {"SACPMLT", &params.SACPMLT},
        {"SACPEXP", &params.SACPEXP},
        {"PERCFRAC", &params.PERCFRAC},
        {"FRACLOWZ", &params.FRACLOWZ},
        {"IFLWRTE", &params.IFLWRTE},
        {"BASERTE", &params.BASERTE},
        {"QB_POWR", &params.QB_POWR},
        {"QB_PRMS", &params.QB_PRMS},
        {"QBRATE_2A", &params.QBRATE_2A},
        {"QBRATE_2B", &params.QBRATE_2B},
        {"SAREAMAX", &params.SAREAMAX},
        {"AXV_BEXP", &params.AXV_BEXP},
        {"LOGLAMB", &params.LOGLAMB},
        {"TISHAPE", &params.TISHAPE},
        {"TIMEDELAY", &params.TIMEDELAY},
        {"MBASE", &params.MBASE},
        {"MFMAX", &params.MFMAX},
        {"MFMIN", &params.MFMIN},
        {"PXTEMP", &params.PXTEMP}
    };
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '(' || line[0] == '*' || line[0] == '!') continue;
        
        std::istringstream iss(line);
        std::string fit_flag, stoch_flag;
        Real default_val, lower, upper;
        
        if (!(iss >> fit_flag >> stoch_flag >> default_val)) continue;
        
        // Find parameter name in the line
        for (const auto& [name, ptr] : param_map) {
            if (line.find(name) != std::string::npos) {
                *ptr = default_val;
                break;
            }
        }
    }
    
    return params;
}

Parameters fortran_to_dfuse_params(const FortranParams& fp) {
    Parameters p;
    p.S1_max = fp.MAXWATR_1;
    p.S2_max = fp.MAXWATR_2;
    p.f_tens = fp.FRACTEN;
    p.f_rchr = fp.FRCHZNE;
    p.f_base = fp.FPRIMQB;
    p.r1 = fp.RTFRAC1;
    p.ku = fp.PERCRTE;
    p.c = fp.PERCEXP;
    p.alpha = fp.SACPMLT;
    p.psi = fp.SACPEXP;
    p.kappa = fp.PERCFRAC;
    p.ki = fp.IFLWRTE;
    p.ks = fp.BASERTE;
    p.n = fp.QB_POWR;
    p.v = fp.QB_PRMS;
    p.v_A = fp.QBRATE_2A;
    p.v_B = fp.QBRATE_2B;
    p.Ac_max = fp.SAREAMAX;
    p.b = fp.AXV_BEXP;
    p.lambda_n = fp.LOGLAMB;
    p.chi = fp.TISHAPE;
    p.mu_t = fp.TIMEDELAY;
    p.T_rain = fp.PXTEMP;
    p.melt_rate = (fp.MFMAX + fp.MFMIN) / 2.0;
    p.compute_derived();
    return p;
}

// ============================================================================
// NETCDF I/O
// ============================================================================

#ifdef DFUSE_USE_NETCDF

struct ForcingData {
    std::vector<double> time;
    std::vector<Real> precip;
    std::vector<Real> pet;
    std::vector<Real> temp;
    std::vector<Real> q_obs;
    std::string time_units;
};

ForcingData read_forcing_netcdf(const std::string& path) {
    ForcingData data;
    
    NcFile file(path, NcFile::read);
    
    // Get dimensions
    NcDim time_dim = file.getDim("time");
    size_t n_time = time_dim.getSize();
    
    // Read time
    NcVar time_var = file.getVar("time");
    data.time.resize(n_time);
    time_var.getVar(data.time.data());
    
    // Get time units
    NcVarAtt units_att = time_var.getAtt("units");
    units_att.getValues(data.time_units);
    
    // Read forcing variables
    auto read_var = [&](const std::string& name) -> std::vector<Real> {
        std::vector<Real> result(n_time);
        NcVar var = file.getVar(name);
        std::vector<float> temp(n_time);
        var.getVar(temp.data());
        for (size_t i = 0; i < n_time; ++i) {
            result[i] = static_cast<Real>(temp[i]);
        }
        return result;
    };
    
    data.precip = read_var("pr");
    data.pet = read_var("pet");
    data.temp = read_var("temp");
    
    // Try to read observed discharge
    try {
        data.q_obs = read_var("q_obs");
    } catch (...) {
        data.q_obs.resize(n_time, std::numeric_limits<Real>::quiet_NaN());
    }
    
    return data;
}

void write_output_netcdf(
    const std::string& path,
    const std::vector<double>& time,
    const std::vector<Real>& runoff,
    const std::string& time_units
) {
    NcFile file(path, NcFile::replace);
    
    // Create dimensions
    NcDim time_dim = file.addDim("time", time.size());
    
    // Create variables
    NcVar time_var = file.addVar("time", ncDouble, time_dim);
    time_var.putAtt("units", time_units);
    time_var.putAtt("long_name", "time");
    time_var.putVar(time.data());
    
    NcVar q_var = file.addVar("q_routed", ncFloat, time_dim);
    q_var.putAtt("units", "mm/day");
    q_var.putAtt("long_name", "Simulated discharge");
    std::vector<float> runoff_f(runoff.begin(), runoff.end());
    q_var.putVar(runoff_f.data());
    
    // Global attributes
    file.putAtt("title", "dFUSE model output");
    file.putAtt("source", "dFUSE - Differentiable FUSE");
}

#else

// Stub implementations when NetCDF not available
struct ForcingData {
    std::vector<double> time;
    std::vector<Real> precip;
    std::vector<Real> pet;
    std::vector<Real> temp;
    std::vector<Real> q_obs;
    std::string time_units;
};

ForcingData read_forcing_netcdf(const std::string& path) {
    throw std::runtime_error("NetCDF support not compiled. Rebuild with -DDFUSE_USE_NETCDF=ON");
}

void write_output_netcdf(
    const std::string& path,
    const std::vector<double>& time,
    const std::vector<Real>& runoff,
    const std::string& time_units
) {
    throw std::runtime_error("NetCDF support not compiled. Rebuild with -DDFUSE_USE_NETCDF=ON");
}

#endif

// ============================================================================
// MAIN CLI
// ============================================================================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <fileManager> <basinID> <runMode>\n";
    std::cerr << "\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  fileManager  Path to FUSE file manager (fm_*.txt)\n";
    std::cerr << "  basinID      Basin identifier\n";
    std::cerr << "  runMode      run_def (default parameters) or run_pre (preset)\n";
    std::cerr << "\n";
    std::cerr << "Example:\n";
    std::cerr << "  " << prog << " fm_catch.txt Klondike_Bonanza_Creek run_def\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string fm_path = argv[1];
    std::string basin_id = argv[2];
    std::string run_mode = argv[3];
    
    if (run_mode != "run_def" && run_mode != "run_pre") {
        std::cerr << "Error: runMode must be 'run_def' or 'run_pre'\n";
        return 1;
    }
    
    try {
        std::cout << "dFUSE v0.2.9\n";
        std::cout << std::string(60, '=') << "\n";
        
        // Parse file manager
        std::cout << "Parsing file manager...\n";
        FileManager fm = parse_file_manager(fm_path);
        
        fs::path input_path(fm.input_path);
        fs::path output_path(fm.output_path);
        fs::path setngs_path(fm.setngs_path);
        
        // Create output directory
        fs::create_directories(output_path);
        
        std::cout << "Basin: " << basin_id << "\n";
        std::cout << "Simulation: " << fm.date_start_sim << " to " << fm.date_end_sim << "\n";
        
        // Parse decisions
        std::cout << "\nParsing model decisions...\n";
        fs::path decisions_path = setngs_path / fm.m_decisions;
        ModelDecisions decisions = parse_decisions(decisions_path.string());
        ModelConfig config = decisions_to_config(decisions);
        
        std::cout << "  Upper layer: " << decisions.arch1 << "\n";
        std::cout << "  Lower layer: " << decisions.arch2 << "\n";
        std::cout << "  Surface runoff: " << decisions.qsurf << "\n";
        std::cout << "  Snow: " << decisions.snowmod << "\n";
        
        // Parse parameters
        std::cout << "\nParsing parameters...\n";
        fs::path constraints_path = setngs_path / fm.constraints;
        FortranParams fp = parse_constraints(constraints_path.string());
        Parameters params = fortran_to_dfuse_params(fp);
        
        std::cout << "  S1_max: " << params.S1_max << " mm\n";
        std::cout << "  S2_max: " << params.S2_max << " mm\n";
        std::cout << "  ks: " << params.ks << " mm/day\n";
        std::cout << "  T_rain: " << params.T_rain << " °C\n";
        std::cout << "  melt_rate: " << params.melt_rate << " mm/°C/day\n";
        
        // Load forcing
        std::cout << "\nLoading forcing data...\n";
        fs::path forcing_path = input_path / (basin_id + fm.suffix_forcing);
        ForcingData forcing = read_forcing_netcdf(forcing_path.string());
        
        size_t n_timesteps = forcing.time.size();
        std::cout << "  Timesteps: " << n_timesteps << "\n";
        
        // Compute forcing statistics
        Real precip_mean = 0, temp_mean = 0;
        for (size_t i = 0; i < n_timesteps; ++i) {
            precip_mean += forcing.precip[i];
            temp_mean += forcing.temp[i];
        }
        precip_mean /= n_timesteps;
        temp_mean /= n_timesteps;
        std::cout << "  Precip: " << precip_mean << " mm/day (mean)\n";
        std::cout << "  Temp: " << temp_mean << " °C (mean)\n";
        
        // Initialize state
        State state;
        state.S1 = params.S1_max * 0.5;
        state.S2 = params.S2_max * 0.5;
        state.SWE = 0.0;
        
        // Run model
        std::cout << "\nRunning dFUSE...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<Real> runoff(n_timesteps);
        Flux flux;
        
        for (size_t t = 0; t < n_timesteps; ++t) {
            Forcing f(forcing.precip[t], forcing.pet[t], forcing.temp[t]);
            fuse_step(state, f, params, config, 1.0, flux);
            runoff[t] = flux.q_total;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        // Compute runoff statistics
        Real runoff_mean = 0, runoff_max = 0;
        for (size_t i = 0; i < n_timesteps; ++i) {
            runoff_mean += runoff[i];
            runoff_max = std::max(runoff_max, runoff[i]);
        }
        runoff_mean /= n_timesteps;
        
        std::cout << "  Completed in " << elapsed << " seconds\n";
        std::cout << "  Runoff: mean=" << runoff_mean << ", max=" << runoff_max << " mm/day\n";
        
        // Write output
        fs::path output_file = output_path / (basin_id + "_" + fm.fmodel_id + "_dfuse.nc");
        write_output_netcdf(output_file.string(), forcing.time, runoff, forcing.time_units);
        std::cout << "\nOutput saved: " << output_file << "\n";
        
        // Compute metrics if observed data available
        size_t n_valid = 0;
        Real ss_res = 0, ss_tot = 0, q_obs_mean = 0;
        for (size_t i = 0; i < n_timesteps; ++i) {
            if (!std::isnan(forcing.q_obs[i]) && forcing.q_obs[i] > -9000) {
                q_obs_mean += forcing.q_obs[i];
                n_valid++;
            }
        }
        
        if (n_valid > 0) {
            q_obs_mean /= n_valid;
            for (size_t i = 0; i < n_timesteps; ++i) {
                if (!std::isnan(forcing.q_obs[i]) && forcing.q_obs[i] > -9000) {
                    ss_res += (runoff[i] - forcing.q_obs[i]) * (runoff[i] - forcing.q_obs[i]);
                    ss_tot += (forcing.q_obs[i] - q_obs_mean) * (forcing.q_obs[i] - q_obs_mean);
                }
            }
            Real nse = 1.0 - ss_res / ss_tot;
            std::cout << "\nPerformance vs observed:\n";
            std::cout << "  NSE: " << nse << "\n";
        }
        
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "dFUSE completed successfully\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
