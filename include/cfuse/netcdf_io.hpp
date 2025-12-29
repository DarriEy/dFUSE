/**
 * @file netcdf_io.hpp
 * @brief NetCDF I/O for FUSE-standard input/output files
 * 
 * Supports reading/writing:
 * - Forcing data (precipitation, PET, temperature)
 * - Model states (S1, S2, SWE, etc.)
 * - Flux outputs (runoff, ET, baseflow, etc.)
 * - Parameters (calibrated or default)
 * 
 * File format follows FUSE conventions from Clark et al. (2008)
 * and is compatible with SUMMA/mizuRoute file structures.
 */

#ifndef DFUSE_NETCDF_IO_HPP
#define DFUSE_NETCDF_IO_HPP

#include "state.hpp"
#include "config.hpp"

#include <string>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef DFUSE_USE_NETCDF
#include <netcdf.h>
#endif

namespace dfuse {
namespace io {

// ============================================================================
// ERROR HANDLING
// ============================================================================

class NetCDFError : public std::runtime_error {
public:
    explicit NetCDFError(const std::string& msg) : std::runtime_error(msg) {}
    
#ifdef DFUSE_USE_NETCDF
    static void check(int status, const std::string& context) {
        if (status != NC_NOERR) {
            throw NetCDFError(context + ": " + nc_strerror(status));
        }
    }
#endif
};

// ============================================================================
// TIME UTILITIES
// ============================================================================

/**
 * @brief Convert days since reference to datetime string
 */
inline std::string days_to_datetime(double days, int ref_year = 1970, int ref_month = 1, int ref_day = 1) {
    // Simple conversion - days since reference date
    time_t ref_time = 0;
    struct tm ref_tm = {};
    ref_tm.tm_year = ref_year - 1900;
    ref_tm.tm_mon = ref_month - 1;
    ref_tm.tm_mday = ref_day;
    ref_time = mktime(&ref_tm);
    
    time_t target_time = ref_time + static_cast<time_t>(days * 86400);
    struct tm* target_tm = gmtime(&target_time);
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(4) << (target_tm->tm_year + 1900) << "-"
        << std::setw(2) << (target_tm->tm_mon + 1) << "-"
        << std::setw(2) << target_tm->tm_mday;
    return oss.str();
}

/**
 * @brief Parse datetime string to days since reference
 */
inline double datetime_to_days(const std::string& datetime, int ref_year = 1970, int ref_month = 1, int ref_day = 1) {
    int year, month, day;
    char sep1, sep2;
    std::istringstream iss(datetime);
    iss >> year >> sep1 >> month >> sep2 >> day;
    
    struct tm ref_tm = {};
    ref_tm.tm_year = ref_year - 1900;
    ref_tm.tm_mon = ref_month - 1;
    ref_tm.tm_mday = ref_day;
    time_t ref_time = mktime(&ref_tm);
    
    struct tm target_tm = {};
    target_tm.tm_year = year - 1900;
    target_tm.tm_mon = month - 1;
    target_tm.tm_mday = day;
    time_t target_time = mktime(&target_tm);
    
    return difftime(target_time, ref_time) / 86400.0;
}

// ============================================================================
// FORCING DATA STRUCTURES
// ============================================================================

/**
 * @brief Container for forcing time series data
 */
struct ForcingData {
    std::vector<double> time;       ///< Time (days since reference)
    std::vector<Real> precip;       ///< Precipitation (mm/day)
    std::vector<Real> pet;          ///< Potential ET (mm/day)
    std::vector<Real> temp;         ///< Air temperature (°C)
    
    // Optional additional forcing
    std::vector<Real> shortwave;    ///< Shortwave radiation (W/m²)
    std::vector<Real> longwave;     ///< Longwave radiation (W/m²)
    std::vector<Real> wind;         ///< Wind speed (m/s)
    std::vector<Real> humidity;     ///< Relative humidity (-)
    std::vector<Real> pressure;     ///< Air pressure (Pa)
    
    // Metadata
    std::string time_units;         ///< e.g., "days since 1970-01-01"
    std::string basin_id;
    double latitude = 0;
    double longitude = 0;
    double elevation = 0;
    double area_km2 = 0;
    
    size_t size() const { return time.size(); }
    bool empty() const { return time.empty(); }
    
    Forcing get_forcing(size_t t) const {
        if (t >= size()) throw std::out_of_range("Forcing index out of range");
        return Forcing{precip[t], pet[t], temp[t]};
    }
};

/**
 * @brief Container for output flux time series
 */
struct OutputData {
    std::vector<double> time;
    
    // State trajectories
    std::vector<Real> S1;
    std::vector<Real> S2;
    std::vector<Real> SWE;
    
    // Flux time series
    std::vector<Real> q_total;      ///< Total runoff
    std::vector<Real> qsx;          ///< Surface runoff
    std::vector<Real> qif;          ///< Interflow
    std::vector<Real> qb;           ///< Baseflow
    std::vector<Real> q12;          ///< Percolation
    std::vector<Real> e_total;      ///< Total ET
    std::vector<Real> e1;           ///< Upper layer ET
    std::vector<Real> e2;           ///< Lower layer ET
    std::vector<Real> rain;         ///< Rainfall
    std::vector<Real> melt;         ///< Snowmelt
    
    // Routed streamflow (optional)
    std::vector<Real> streamflow;
    
    // Metadata
    std::string basin_id;
    std::string model_config;
    
    void resize(size_t n) {
        time.resize(n);
        S1.resize(n); S2.resize(n); SWE.resize(n);
        q_total.resize(n); qsx.resize(n); qif.resize(n); qb.resize(n);
        q12.resize(n); e_total.resize(n); e1.resize(n); e2.resize(n);
        rain.resize(n); melt.resize(n);
    }
    
    void record(size_t t, double time_val, const State& state, const Flux& flux) {
        if (t >= time.size()) return;
        time[t] = time_val;
        S1[t] = state.S1;
        S2[t] = state.S2;
        SWE[t] = state.SWE;
        q_total[t] = flux.q_total;
        qsx[t] = flux.qsx;
        qif[t] = flux.qif;
        qb[t] = flux.qb;
        q12[t] = flux.q12;
        e_total[t] = flux.e_total;
        e1[t] = flux.e1;
        e2[t] = flux.e2;
        rain[t] = flux.rain;
        melt[t] = flux.melt;
    }
};

// ============================================================================
// CSV I/O (FALLBACK WHEN NETCDF NOT AVAILABLE)
// ============================================================================

/**
 * @brief Read forcing data from CSV file
 * 
 * Expected format:
 * time,precip,pet,temp
 * 0,5.2,3.1,15.0
 * 1,0.0,3.5,18.2
 * ...
 */
inline ForcingData read_forcing_csv(const std::string& filename) {
    ForcingData data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open forcing file: " + filename);
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double t;
        Real p, pet, temp;
        char comma;
        
        if (iss >> t >> comma >> p >> comma >> pet >> comma >> temp) {
            data.time.push_back(t);
            data.precip.push_back(p);
            data.pet.push_back(pet);
            data.temp.push_back(temp);
        }
    }
    
    return data;
}

/**
 * @brief Write output data to CSV file
 */
inline void write_output_csv(const std::string& filename, const OutputData& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }
    
    // Header
    file << "time,S1,S2,SWE,q_total,qsx,qif,qb,q12,e_total,e1,e2,rain,melt\n";
    
    // Data
    file << std::fixed << std::setprecision(6);
    for (size_t t = 0; t < data.time.size(); ++t) {
        file << data.time[t] << ","
             << data.S1[t] << "," << data.S2[t] << "," << data.SWE[t] << ","
             << data.q_total[t] << "," << data.qsx[t] << "," << data.qif[t] << ","
             << data.qb[t] << "," << data.q12[t] << ","
             << data.e_total[t] << "," << data.e1[t] << "," << data.e2[t] << ","
             << data.rain[t] << "," << data.melt[t] << "\n";
    }
}

/**
 * @brief Write parameters to CSV file
 */
inline void write_parameters_csv(const std::string& filename, const Parameters& params) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open parameter file: " + filename);
    }
    
    file << std::fixed << std::setprecision(8);
    file << "parameter,value\n";
    file << "S1_max," << params.S1_max << "\n";
    file << "S2_max," << params.S2_max << "\n";
    file << "f_tens," << params.f_tens << "\n";
    file << "f_rchr," << params.f_rchr << "\n";
    file << "f_base," << params.f_base << "\n";
    file << "r1," << params.r1 << "\n";
    file << "ku," << params.ku << "\n";
    file << "c," << params.c << "\n";
    file << "alpha," << params.alpha << "\n";
    file << "psi," << params.psi << "\n";
    file << "kappa," << params.kappa << "\n";
    file << "ki," << params.ki << "\n";
    file << "ks," << params.ks << "\n";
    file << "n," << params.n << "\n";
    file << "v," << params.v << "\n";
    file << "v_A," << params.v_A << "\n";
    file << "v_B," << params.v_B << "\n";
    file << "Ac_max," << params.Ac_max << "\n";
    file << "b," << params.b << "\n";
    file << "lambda," << params.lambda << "\n";
    file << "chi," << params.chi << "\n";
    file << "mu_t," << params.mu_t << "\n";
    file << "shape_t," << params.shape_t << "\n";
    file << "T_rain," << params.T_rain << "\n";
    file << "T_melt," << params.T_melt << "\n";
    file << "melt_rate," << params.melt_rate << "\n";
}

/**
 * @brief Read parameters from CSV file
 */
inline Parameters read_parameters_csv(const std::string& filename) {
    Parameters params;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open parameter file: " + filename);
    }
    
    std::string line;
    std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string name;
        char comma;
        Real value;
        
        if (std::getline(iss, name, ',') && iss >> value) {
            if (name == "S1_max") params.S1_max = value;
            else if (name == "S2_max") params.S2_max = value;
            else if (name == "f_tens") params.f_tens = value;
            else if (name == "f_rchr") params.f_rchr = value;
            else if (name == "f_base") params.f_base = value;
            else if (name == "r1") params.r1 = value;
            else if (name == "ku") params.ku = value;
            else if (name == "c") params.c = value;
            else if (name == "alpha") params.alpha = value;
            else if (name == "psi") params.psi = value;
            else if (name == "kappa") params.kappa = value;
            else if (name == "ki") params.ki = value;
            else if (name == "ks") params.ks = value;
            else if (name == "n") params.n = value;
            else if (name == "v") params.v = value;
            else if (name == "v_A") params.v_A = value;
            else if (name == "v_B") params.v_B = value;
            else if (name == "Ac_max") params.Ac_max = value;
            else if (name == "b") params.b = value;
            else if (name == "lambda") params.lambda = value;
            else if (name == "chi") params.chi = value;
            else if (name == "mu_t") params.mu_t = value;
            else if (name == "shape_t") params.shape_t = value;
            else if (name == "T_rain") params.T_rain = value;
            else if (name == "T_melt") params.T_melt = value;
            else if (name == "melt_rate") params.melt_rate = value;
        }
    }
    
    params.compute_derived();
    return params;
}

// ============================================================================
// NETCDF I/O (WHEN AVAILABLE)
// ============================================================================

#ifdef DFUSE_USE_NETCDF

/**
 * @brief Read forcing data from NetCDF file
 * 
 * Supports FUSE/SUMMA-style forcing files with dimensions:
 * - time: unlimited
 * - hru: number of hydrologic response units (optional)
 */
inline ForcingData read_forcing_netcdf(const std::string& filename, int hru_index = 0) {
    ForcingData data;
    
    int ncid;
    NetCDFError::check(nc_open(filename.c_str(), NC_NOWRITE, &ncid), "Opening file");
    
    // Get time dimension
    int time_dimid;
    size_t n_time;
    NetCDFError::check(nc_inq_dimid(ncid, "time", &time_dimid), "Finding time dimension");
    NetCDFError::check(nc_inq_dimlen(ncid, time_dimid, &n_time), "Getting time length");
    
    // Read time variable
    int time_varid;
    data.time.resize(n_time);
    NetCDFError::check(nc_inq_varid(ncid, "time", &time_varid), "Finding time variable");
    NetCDFError::check(nc_get_var_double(ncid, time_varid, data.time.data()), "Reading time");
    
    // Get time units attribute
    char units[256];
    if (nc_get_att_text(ncid, time_varid, "units", units) == NC_NOERR) {
        data.time_units = std::string(units);
    }
    
    // Helper to read variable (handles both 1D and 2D cases)
    auto read_var = [&](const char* name, std::vector<Real>& vec) {
        int varid;
        if (nc_inq_varid(ncid, name, &varid) != NC_NOERR) return false;
        
        int ndims;
        nc_inq_varndims(ncid, varid, &ndims);
        
        vec.resize(n_time);
        
        if (ndims == 1) {
            // Simple 1D variable
            std::vector<float> temp(n_time);
            nc_get_var_float(ncid, varid, temp.data());
            for (size_t i = 0; i < n_time; ++i) vec[i] = temp[i];
        } else {
            // 2D variable [time, hru]
            std::vector<float> temp(n_time);
            size_t start[2] = {0, static_cast<size_t>(hru_index)};
            size_t count[2] = {n_time, 1};
            nc_get_vara_float(ncid, varid, start, count, temp.data());
            for (size_t i = 0; i < n_time; ++i) vec[i] = temp[i];
        }
        return true;
    };
    
    // Read forcing variables (try various naming conventions)
    if (!read_var("pptrate", data.precip)) {
        if (!read_var("precip", data.precip)) {
            read_var("pr", data.precip);
        }
    }
    
    if (!read_var("pet", data.pet)) {
        if (!read_var("potevap", data.pet)) {
            read_var("PET", data.pet);
        }
    }
    
    if (!read_var("airtemp", data.temp)) {
        if (!read_var("temp", data.temp)) {
            if (!read_var("tas", data.temp)) {
                read_var("t2m", data.temp);
            }
        }
    }
    
    // Optional variables
    read_var("SWRadAtm", data.shortwave);
    read_var("LWRadAtm", data.longwave);
    read_var("windspd", data.wind);
    read_var("spechum", data.humidity);
    read_var("airpres", data.pressure);
    
    nc_close(ncid);
    return data;
}

/**
 * @brief Write output data to NetCDF file
 */
inline void write_output_netcdf(const std::string& filename, const OutputData& data,
                                 const std::string& time_units = "days since 1970-01-01") {
    int ncid;
    NetCDFError::check(nc_create(filename.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid), "Creating file");
    
    // Define dimensions
    int time_dimid;
    NetCDFError::check(nc_def_dim(ncid, "time", data.time.size(), &time_dimid), "Defining time dim");
    
    // Define variables
    int time_varid, s1_varid, s2_varid, swe_varid;
    int q_varid, qsx_varid, qif_varid, qb_varid, q12_varid;
    int et_varid, e1_varid, e2_varid, rain_varid, melt_varid;
    
    NetCDFError::check(nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid), "Def time");
    NetCDFError::check(nc_def_var(ncid, "S1", NC_FLOAT, 1, &time_dimid, &s1_varid), "Def S1");
    NetCDFError::check(nc_def_var(ncid, "S2", NC_FLOAT, 1, &time_dimid, &s2_varid), "Def S2");
    NetCDFError::check(nc_def_var(ncid, "SWE", NC_FLOAT, 1, &time_dimid, &swe_varid), "Def SWE");
    NetCDFError::check(nc_def_var(ncid, "q_total", NC_FLOAT, 1, &time_dimid, &q_varid), "Def q");
    NetCDFError::check(nc_def_var(ncid, "qsx", NC_FLOAT, 1, &time_dimid, &qsx_varid), "Def qsx");
    NetCDFError::check(nc_def_var(ncid, "qif", NC_FLOAT, 1, &time_dimid, &qif_varid), "Def qif");
    NetCDFError::check(nc_def_var(ncid, "qb", NC_FLOAT, 1, &time_dimid, &qb_varid), "Def qb");
    NetCDFError::check(nc_def_var(ncid, "q12", NC_FLOAT, 1, &time_dimid, &q12_varid), "Def q12");
    NetCDFError::check(nc_def_var(ncid, "e_total", NC_FLOAT, 1, &time_dimid, &et_varid), "Def et");
    NetCDFError::check(nc_def_var(ncid, "e1", NC_FLOAT, 1, &time_dimid, &e1_varid), "Def e1");
    NetCDFError::check(nc_def_var(ncid, "e2", NC_FLOAT, 1, &time_dimid, &e2_varid), "Def e2");
    NetCDFError::check(nc_def_var(ncid, "rain", NC_FLOAT, 1, &time_dimid, &rain_varid), "Def rain");
    NetCDFError::check(nc_def_var(ncid, "melt", NC_FLOAT, 1, &time_dimid, &melt_varid), "Def melt");
    
    // Add attributes
    nc_put_att_text(ncid, time_varid, "units", time_units.length(), time_units.c_str());
    nc_put_att_text(ncid, s1_varid, "units", 2, "mm");
    nc_put_att_text(ncid, s2_varid, "units", 2, "mm");
    nc_put_att_text(ncid, swe_varid, "units", 2, "mm");
    nc_put_att_text(ncid, q_varid, "units", 6, "mm/day");
    nc_put_att_text(ncid, qsx_varid, "units", 6, "mm/day");
    nc_put_att_text(ncid, et_varid, "units", 6, "mm/day");
    
    nc_put_att_text(ncid, s1_varid, "long_name", 20, "Upper layer storage");
    nc_put_att_text(ncid, s2_varid, "long_name", 20, "Lower layer storage");
    nc_put_att_text(ncid, q_varid, "long_name", 12, "Total runoff");
    nc_put_att_text(ncid, et_varid, "long_name", 20, "Total evapotranspiration");
    
    // Global attributes
    nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "CF-1.6");
    nc_put_att_text(ncid, NC_GLOBAL, "source", 5, "cFUSE");
    nc_put_att_text(ncid, NC_GLOBAL, "version", 5, "0.2.0");
    if (!data.model_config.empty()) {
        nc_put_att_text(ncid, NC_GLOBAL, "model_config", data.model_config.length(), 
                        data.model_config.c_str());
    }
    
    // End define mode
    NetCDFError::check(nc_enddef(ncid), "End define");
    
    // Write data
    NetCDFError::check(nc_put_var_double(ncid, time_varid, data.time.data()), "Write time");
    
    // Convert Real vectors to float for NetCDF
    auto write_float_var = [&](int varid, const std::vector<Real>& vec) {
        std::vector<float> temp(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) temp[i] = static_cast<float>(vec[i]);
        nc_put_var_float(ncid, varid, temp.data());
    };
    
    write_float_var(s1_varid, data.S1);
    write_float_var(s2_varid, data.S2);
    write_float_var(swe_varid, data.SWE);
    write_float_var(q_varid, data.q_total);
    write_float_var(qsx_varid, data.qsx);
    write_float_var(qif_varid, data.qif);
    write_float_var(qb_varid, data.qb);
    write_float_var(q12_varid, data.q12);
    write_float_var(et_varid, data.e_total);
    write_float_var(e1_varid, data.e1);
    write_float_var(e2_varid, data.e2);
    write_float_var(rain_varid, data.rain);
    write_float_var(melt_varid, data.melt);
    
    nc_close(ncid);
}

#endif // DFUSE_USE_NETCDF

// ============================================================================
// UNIFIED I/O FUNCTIONS (AUTO-DETECT FORMAT)
// ============================================================================

/**
 * @brief Read forcing data (auto-detects format from extension)
 */
inline ForcingData read_forcing(const std::string& filename, int hru_index = 0) {
    // Check extension
    size_t dot_pos = filename.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string ext = filename.substr(dot_pos);
        if (ext == ".csv" || ext == ".txt") {
            return read_forcing_csv(filename);
        }
#ifdef DFUSE_USE_NETCDF
        if (ext == ".nc" || ext == ".nc4") {
            return read_forcing_netcdf(filename, hru_index);
        }
#endif
    }
    
    // Default to CSV
    return read_forcing_csv(filename);
}

/**
 * @brief Write output data (auto-detects format from extension)
 */
inline void write_output(const std::string& filename, const OutputData& data) {
    size_t dot_pos = filename.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string ext = filename.substr(dot_pos);
#ifdef DFUSE_USE_NETCDF
        if (ext == ".nc" || ext == ".nc4") {
            write_output_netcdf(filename, data);
            return;
        }
#endif
    }
    
    // Default to CSV
    write_output_csv(filename, data);
}

} // namespace io
} // namespace dfuse

#endif // DFUSE_NETCDF_IO_HPP
