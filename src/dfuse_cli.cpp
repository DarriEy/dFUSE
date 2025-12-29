/**
 * @file dfuse_cli.cpp
 * @brief cFUSE Command Line Interface - Drop-in replacement for Fortran FUSE
 * 
 * Usage:
 *   cfuse <fileManager> <basinID> <runMode>
 * 
 * Arguments:
 *   fileManager: Path to FUSE file manager (fm_*.txt)
 *   basinID: Basin identifier
 *   runMode: run_def (default parameters) or run_pre (preset)
 */

#include "cfuse/dfuse.hpp"
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
 #include <netcdf.h>
 using namespace netCDF;
 #endif
 
 #ifdef _OPENMP
 #include <omp.h>
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
     else if (dec.arch2 == "topmdexp_2") config.lower_arch = LowerLayerArch::SINGLE_NOEVAP;  // TOPMODEL uses infinite reservoir
     
     // Baseflow
     if (dec.arch2 == "fixedsiz_2") config.baseflow = BaseflowType::LINEAR;  // Fixed size with linear drainage
     else if (dec.arch2 == "unlimfrc_2") config.baseflow = BaseflowType::LINEAR;
     else if (dec.arch2 == "unlimpow_2") config.baseflow = BaseflowType::NONLINEAR;
     else if (dec.arch2 == "tens2pll_2") config.baseflow = BaseflowType::PARALLEL_LINEAR;
     else if (dec.arch2 == "topmdexp_2") config.baseflow = BaseflowType::TOPMODEL;
     
     // Surface runoff
     if (dec.qsurf == "prms_varnt") config.surface_runoff = SurfaceRunoffType::UZ_LINEAR;
     else if (dec.qsurf == "arno_x_vic") config.surface_runoff = SurfaceRunoffType::UZ_PARETO;
     else if (dec.qsurf == "tmdl_param") config.surface_runoff = SurfaceRunoffType::LZ_GAMMA;
     
     // Percolation
     // perc_f2sat = "field capacity to saturation" = FREE storage only
     // perc_w2sat = "wilting point to saturation" = TOTAL storage
     if (dec.qperc == "perc_f2sat") config.percolation = PercolationType::FREE_STORAGE;
     else if (dec.qperc == "perc_w2sat") config.percolation = PercolationType::TOTAL_STORAGE;
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
 
Parameters fortran_to_cfuse_params(const FortranParams& fp) {
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
 
 struct DistributedForcingData {
     std::vector<double> time;
     std::vector<std::vector<Real>> precip;  // [hru][time]
     std::vector<std::vector<Real>> pet;
     std::vector<std::vector<Real>> temp;
     std::vector<Real> q_obs;
     std::vector<Real> hru_area;             // Area of each HRU (km²)
     std::string time_units;
     size_t n_hru;
     size_t n_time;
     bool is_distributed;
 };
 
 // Check if file is distributed (has hru dimension) and get metadata
 DistributedForcingData read_forcing_distributed(const std::string& path) {
     DistributedForcingData data;
     
     NcFile file(path, NcFile::read);
     
     // Get time dimension
     NcDim time_dim = file.getDim("time");
     data.n_time = time_dim.getSize();
     
     // Check for HRU dimension - try various names used in different FUSE setups
     // FUSE often uses latitude as the HRU dimension with longitude=1
     NcDim hru_dim;
     std::string hru_dim_name;
     const char* hru_dim_names[] = {"hru", "gru", "HRU", "GRU", "nHRU", "latitude", "lat", "hruid"};
     
     for (const char* name : hru_dim_names) {
         try {
             hru_dim = file.getDim(name);
             if (!hru_dim.isNull()) {
                 data.n_hru = hru_dim.getSize();
                 hru_dim_name = name;
                 data.is_distributed = (data.n_hru > 1);
                 break;
             }
         } catch (...) {}
     }
     
     if (hru_dim_name.empty()) {
         data.n_hru = 1;
         data.is_distributed = false;
     }
     
     std::cout << "  HRU dimension: " << (hru_dim_name.empty() ? "none (lumped)" : hru_dim_name) 
               << " (n=" << data.n_hru << ")\n";
     
     // Read time
     NcVar time_var = file.getVar("time");
     data.time.resize(data.n_time);
     time_var.getVar(data.time.data());
     
     // Get time units
     try {
         NcVarAtt units_att = time_var.getAtt("units");
         units_att.getValues(data.time_units);
     } catch (...) {
         data.time_units = "days since 1970-01-01";
     }
     
     // Read HRU areas if available - try various names
     data.hru_area.resize(data.n_hru, 1.0);  // Default to equal areas
     const char* area_names[] = {"hruArea", "HRUarea", "area", "AREA", "gruArea"};
     for (const char* name : area_names) {
         try {
             NcVar area_var = file.getVar(name);
             if (!area_var.isNull()) {
                 std::vector<float> temp(data.n_hru);
                 area_var.getVar(temp.data());
                 for (size_t h = 0; h < data.n_hru; ++h) {
                     data.hru_area[h] = static_cast<Real>(temp[h]);
                 }
                 std::cout << "  Found HRU areas in variable: " << name << "\n";
                 break;
             }
         } catch (...) {}
     }
     
     // Helper to read variable - handles 1D, 2D, and 3D (time, lat, lon) arrays
     auto read_var_nd = [&](const std::string& name) -> std::vector<std::vector<Real>> {
         std::vector<std::vector<Real>> result(data.n_hru);
         for (size_t h = 0; h < data.n_hru; ++h) {
             result[h].resize(data.n_time);
         }
         
         NcVar var = file.getVar(name);
         if (var.isNull()) {
             throw std::runtime_error("Variable not found: " + name);
         }
         
         int ndims = var.getDimCount();
         std::vector<NcDim> dims = var.getDims();
         
         std::cout << "  Reading " << name << ": ndims=" << ndims << ", shape=(";
         for (int d = 0; d < ndims; ++d) {
             std::cout << dims[d].getSize() << (d < ndims-1 ? "," : "");
         }
         std::cout << ")\n";
         
         if (ndims == 1) {
             // 1D variable (time only) - same for all HRUs
             std::vector<float> temp(data.n_time);
             var.getVar(temp.data());
             for (size_t h = 0; h < data.n_hru; ++h) {
                 for (size_t t = 0; t < data.n_time; ++t) {
                     result[h][t] = static_cast<Real>(temp[t]);
                 }
             }
         } else if (ndims == 2) {
             // 2D variable [time, hru] or [hru, time]
             bool time_first = (dims[0].getName() == "time");
             size_t dim0 = dims[0].getSize();
             size_t dim1 = dims[1].getSize();
             
             std::vector<float> temp(dim0 * dim1);
             var.getVar(temp.data());
             
             for (size_t h = 0; h < data.n_hru; ++h) {
                 for (size_t t = 0; t < data.n_time; ++t) {
                     size_t idx = time_first ? (t * data.n_hru + h) : (h * data.n_time + t);
                     result[h][t] = static_cast<Real>(temp[idx]);
                 }
             }
         } else if (ndims == 3) {
             // 3D variable - typically [time, lat, lon] where lat=HRU and lon=1
             // Or could be [time, hru, elevation_band]
             size_t dim0 = dims[0].getSize();  // time
             size_t dim1 = dims[1].getSize();  // lat/hru
             size_t dim2 = dims[2].getSize();  // lon (usually 1)
             
             // Read full array
             std::vector<float> temp(dim0 * dim1 * dim2);
             var.getVar(temp.data());
             
             // Assume [time, hru, lon] layout with lon=1 (Fortran FUSE style)
             // Index: temp[t * dim1 * dim2 + h * dim2 + lon]
             for (size_t h = 0; h < data.n_hru; ++h) {
                 for (size_t t = 0; t < data.n_time; ++t) {
                     // Take lon=0 (first/only longitude)
                     size_t idx = t * dim1 * dim2 + h * dim2 + 0;
                     result[h][t] = static_cast<Real>(temp[idx]);
                 }
             }
         } else {
             throw std::runtime_error("Unsupported number of dimensions for " + name + ": " + std::to_string(ndims));
         }
         
         return result;
     };
     
     // Try various variable names
     auto try_read = [&](const std::vector<std::string>& names) -> std::vector<std::vector<Real>> {
         for (const auto& name : names) {
             try {
                 return read_var_nd(name);
             } catch (const std::exception& e) {
                 // Only print if variable exists but read failed
                 NcVar var = file.getVar(name);
                 if (!var.isNull()) {
                     std::cerr << "  Warning: Failed to read " << name << ": " << e.what() << "\n";
                 }
             }
         }
         throw std::runtime_error("Could not find variable with names: " + names[0]);
     };
     
     data.precip = try_read({"pptrate", "pr", "precip", "prcp"});
     data.pet = try_read({"pet", "potevap", "PET", "etp"});
     data.temp = try_read({"airtemp", "temp", "tas", "t2m", "tmean"});
     
     // Try to read observed discharge (1D - basin total)
     data.q_obs.resize(data.n_time, std::numeric_limits<Real>::quiet_NaN());
     try {
         NcVar qobs_var = file.getVar("q_obs");
         if (!qobs_var.isNull()) {
             int ndims = qobs_var.getDimCount();
             if (ndims == 1) {
                 std::vector<float> temp(data.n_time);
                 qobs_var.getVar(temp.data());
                 for (size_t t = 0; t < data.n_time; ++t) {
                     data.q_obs[t] = static_cast<Real>(temp[t]);
                 }
             }
         }
     } catch (...) {}
     
     return data;
 }
 
 // Legacy function for backward compatibility
 ForcingData read_forcing_netcdf(const std::string& path) {
     auto dist = read_forcing_distributed(path);
     
     ForcingData data;
     data.time = dist.time;
     data.time_units = dist.time_units;
     data.q_obs = dist.q_obs;
     
     // Use first HRU (or only HRU for lumped)
     data.precip.resize(dist.n_time);
     data.pet.resize(dist.n_time);
     data.temp.resize(dist.n_time);
     
     for (size_t t = 0; t < dist.n_time; ++t) {
         data.precip[t] = dist.precip[0][t];
         data.pet[t] = dist.pet[0][t];
         data.temp[t] = dist.temp[0][t];
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
    file.putAtt("title", "cFUSE model output");
    file.putAtt("source", "cFUSE - Differentiable FUSE");
 }
 
 // Write distributed output in mizuRoute-compatible format
 // Uses gru dimension and gruId variable as expected by mizuRoute
 void write_output_netcdf_distributed(
     const std::string& path,
     const std::vector<double>& time,
     const std::vector<std::vector<Real>>& hru_runoff,  // [hru][time]
     const std::string& time_units,
     size_t n_hru
 ) {
     NcFile file(path, NcFile::replace);
     
     size_t n_time = time.size();
     
     // Create dimensions - use 'gru' and 'time' as mizuRoute expects
     NcDim time_dim = file.addDim("time", n_time);
     NcDim gru_dim = file.addDim("gru", n_hru);
     
     // Create time coordinate variable
     NcVar time_var = file.addVar("time", ncDouble, time_dim);
     time_var.putAtt("units", time_units);
     time_var.putAtt("long_name", "time");
     time_var.putAtt("calendar", "standard");
     time_var.putVar(time.data());
     
     // Create gruId variable (HRU identifiers) - mizuRoute needs this
     NcVar gruid_var = file.addVar("gruId", ncInt, gru_dim);
     gruid_var.putAtt("long_name", "GRU identifier");
     gruid_var.putAtt("units", "-");
     // Write HRU IDs as 1, 2, 3, ... n_hru (1-indexed as Fortran FUSE uses)
     std::vector<int> gru_ids(n_hru);
     for (size_t h = 0; h < n_hru; ++h) gru_ids[h] = static_cast<int>(h + 1);
     gruid_var.putVar(gru_ids.data());
     
     // Create runoff variable with 2D shape (time, gru) - standard mizuRoute format
     std::vector<NcDim> dims = {time_dim, gru_dim};
     NcVar q_var = file.addVar("q_routed", ncFloat, dims);
     q_var.putAtt("units", "mm/day");
     q_var.putAtt("long_name", "routed runoff");
     q_var.putAtt("_FillValue", ncFloat, -9999.0f);
     
     // Prepare data in [time, gru] order (C-style, row-major)
     // hru_runoff is [hru][time], we need [time][hru]
     std::vector<float> q_data(n_time * n_hru);
     for (size_t t = 0; t < n_time; ++t) {
         for (size_t h = 0; h < n_hru; ++h) {
             size_t idx = t * n_hru + h;
             q_data[idx] = static_cast<float>(hru_runoff[h][t]);
         }
     }
     q_var.putVar(q_data.data());
     
     // Also write instantaneous runoff (same as routed for now, no internal routing delay)
     NcVar qi_var = file.addVar("q_instnt", ncFloat, dims);
     qi_var.putAtt("units", "mm/day");
     qi_var.putAtt("long_name", "instantaneous runoff");
     qi_var.putAtt("_FillValue", ncFloat, -9999.0f);
     qi_var.putVar(q_data.data());
     
     // Global attributes
    file.putAtt("title", "cFUSE model output");
    file.putAtt("source", "cFUSE - Differentiable FUSE");
     file.putAtt("Conventions", "CF-1.6");
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
 
 struct DistributedForcingData {
     std::vector<double> time;
     std::vector<std::vector<Real>> precip;
     std::vector<std::vector<Real>> pet;
     std::vector<std::vector<Real>> temp;
     std::vector<Real> q_obs;
     std::vector<Real> hru_area;
     std::string time_units;
     size_t n_hru = 0;
     size_t n_time = 0;
     bool is_distributed = false;
 };
 
 ForcingData read_forcing_netcdf(const std::string& path) {
     throw std::runtime_error("NetCDF support not compiled. Rebuild with -DDFUSE_USE_NETCDF=ON");
 }
 
 DistributedForcingData read_forcing_distributed(const std::string& path) {
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
 
 void write_output_netcdf_distributed(
     const std::string& path,
     const std::vector<double>& time,
     const std::vector<std::vector<Real>>& hru_runoff,
     const std::string& time_units,
     size_t n_hru
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
        std::cout << "cFUSE v0.2.9\n";
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
        Parameters params = fortran_to_cfuse_params(fp);
         
         std::cout << "  S1_max: " << params.S1_max << " mm\n";
         std::cout << "  S2_max: " << params.S2_max << " mm\n";
         std::cout << "  ks: " << params.ks << " mm/day\n";
         std::cout << "  T_rain: " << params.T_rain << " °C\n";
         std::cout << "  melt_rate: " << params.melt_rate << " mm/°C/day\n";
         
         // Load forcing data (handles both lumped and distributed)
         std::cout << "\nLoading forcing data...\n";
         fs::path forcing_path = input_path / (basin_id + fm.suffix_forcing);
         
 #ifdef DFUSE_USE_NETCDF
         DistributedForcingData forcing = read_forcing_distributed(forcing_path.string());
 #else
         // Fallback for non-NetCDF build
         ForcingData forcing_1d = read_forcing_netcdf(forcing_path.string());
         DistributedForcingData forcing;
         forcing.n_time = forcing_1d.time.size();
         forcing.n_hru = 1;
         forcing.is_distributed = false;
         forcing.time = forcing_1d.time;
         forcing.time_units = forcing_1d.time_units;
         forcing.q_obs = forcing_1d.q_obs;
         forcing.hru_area = {1.0};
         forcing.precip = {forcing_1d.precip};
         forcing.pet = {forcing_1d.pet};
         forcing.temp = {forcing_1d.temp};
 #endif
         
         size_t n_timesteps = forcing.n_time;
         size_t n_hru = forcing.n_hru;
         
         std::cout << "  Timesteps: " << n_timesteps << "\n";
         std::cout << "  HRUs: " << n_hru << (forcing.is_distributed ? " (distributed mode)" : " (lumped mode)") << "\n";
         
         // Compute total basin area for area-weighting
         Real total_area = 0;
         for (size_t h = 0; h < n_hru; ++h) {
             total_area += forcing.hru_area[h];
         }
         std::cout << "  Total area: " << total_area << " (arbitrary units)\n";
         
         // Compute forcing statistics (area-weighted mean)
         Real precip_mean = 0, temp_mean = 0;
         for (size_t h = 0; h < n_hru; ++h) {
             Real weight = forcing.hru_area[h] / total_area;
             for (size_t t = 0; t < n_timesteps; ++t) {
                 precip_mean += forcing.precip[h][t] * weight;
                 temp_mean += forcing.temp[h][t] * weight;
             }
         }
         precip_mean /= n_timesteps;
         temp_mean /= n_timesteps;
         std::cout << "  Precip: " << precip_mean << " mm/day (area-weighted mean)\n";
         std::cout << "  Temp: " << temp_mean << " °C (area-weighted mean)\n";
         
         // Run model
        std::cout << "\nRunning cFUSE";
         if (n_hru > 1) {
 #ifdef _OPENMP
             std::cout << " with OpenMP (" << omp_get_max_threads() << " threads)";
 #endif
         }
         std::cout << "...\n";
         
         auto start_time = std::chrono::high_resolution_clock::now();
         
         // Allocate per-HRU runoff arrays
         std::vector<std::vector<Real>> hru_runoff(n_hru);
         for (size_t h = 0; h < n_hru; ++h) {
             hru_runoff[h].resize(n_timesteps);
         }
         
         // Initialize state template - will be copied for each HRU
         auto init_state = [&](State& state) {
             constexpr Real fracstate0 = 0.25;
             state.SWE = 0.0;
             
             // Upper layer initialization based on architecture
             switch (config.upper_arch) {
                 case UpperLayerArch::SINGLE_STATE:
                     state.S1 = params.S1_max * fracstate0;
                     break;
                 case UpperLayerArch::TENSION_FREE:
                     state.S1_T = params.S1_T_max * fracstate0;
                     state.S1_F = params.S1_F_max * fracstate0;
                     break;
                 case UpperLayerArch::TENSION2_FREE:
                     state.S1_TA = params.S1_TA_max * fracstate0;
                     state.S1_TB = params.S1_TB_max * fracstate0;
                     state.S1_F = params.S1_F_max * fracstate0;
                     break;
             }
             
             // Lower layer initialization based on architecture
             switch (config.lower_arch) {
                 case LowerLayerArch::SINGLE_NOEVAP:
                 case LowerLayerArch::SINGLE_EVAP:
                     state.S2 = params.S2_max * fracstate0;
                     break;
                 case LowerLayerArch::TENSION_2RESERV:
                     state.S2_T = params.S2_T_max * fracstate0;
                     state.S2_FA = params.S2_FA_max * fracstate0;
                     state.S2_FB = params.S2_FB_max * fracstate0;
                     break;
             }
             
             state.sync_derived(config);
         };
         
         // Run FUSE for each HRU (parallelized with OpenMP)
         #pragma omp parallel for schedule(dynamic) if(n_hru > 1)
         for (size_t h = 0; h < n_hru; ++h) {
             State state;
             init_state(state);
             Flux flux;
             
             for (size_t t = 0; t < n_timesteps; ++t) {
                 Forcing f(forcing.precip[h][t], forcing.pet[h][t], forcing.temp[h][t]);
                 fuse_step(state, f, params, config, 1.0, flux);
                 hru_runoff[h][t] = flux.q_total;
             }
         }
         
         // Area-weighted aggregation of runoff
         std::vector<Real> runoff(n_timesteps, 0.0);
         for (size_t t = 0; t < n_timesteps; ++t) {
             for (size_t h = 0; h < n_hru; ++h) {
                 runoff[t] += hru_runoff[h][t] * (forcing.hru_area[h] / total_area);
             }
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
         
         std::cout << "  Completed in " << elapsed << " seconds";
         if (n_hru > 1) {
             std::cout << " (" << (n_timesteps * n_hru) / elapsed / 1e6 << " M timestep-HRU/s)";
         }
         std::cout << "\n";
         std::cout << "  Runoff: mean=" << runoff_mean << ", max=" << runoff_max << " mm/day (area-weighted)\n";
         
         // Write output
        fs::path output_file = output_path / (basin_id + "_" + fm.fmodel_id + "_cfuse.nc");
         
         if (forcing.is_distributed) {
             // Write per-HRU output in Fortran FUSE format (time, latitude, longitude)
             // This format is compatible with mizuRoute
             write_output_netcdf_distributed(output_file.string(), forcing.time, hru_runoff, 
                                             forcing.time_units, n_hru);
             std::cout << "\nOutput saved (distributed, mizuRoute-compatible): " << output_file << "\n";
         } else {
             // Write lumped output
             write_output_netcdf(output_file.string(), forcing.time, runoff, forcing.time_units);
             std::cout << "\nOutput saved (lumped): " << output_file << "\n";
         }
         
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
        std::cout << "cFUSE completed successfully\n";
         
         return 0;
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << "\n";
         return 1;
     }
 }
