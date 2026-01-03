# ===============================================================================
# GRIB Processing Pipeline
# Author: Simon Horton
# Description: Processes GRIB files for NWP archives using array jobs for parallel processing
# ===============================================================================

# ===============================================================================
# 1. Configuration and Setup
# ===============================================================================

# Determine if running in test mode or on HPC
test_mode <- Sys.info()["nodename"] %in% c("SIMON-ASUS", "Simon-ASUS")

# Parse command line arguments for season
if (!test_mode) {
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) != 1) stop("Usage: Rscript gribts.R <season>")
    season <- as.numeric(args[1])
} else {
    season <- 2021
}

# ===============================================================================
# 2. Path Configuration
# ===============================================================================

# Define base paths for test and HPC environments
base_paths <- list(
    test = list(
        wd = "/mnt/c/Users/horto/Data/snowmodels/nwp_archives",
        wgrib = "/home/shorton/wgrib2/wgrib2/wgrib2",
        grib = "/mnt/c/Users/horto/Data/snowmodels/nwp_archives/in",
        out = "/mnt/c/Users/horto/Data/snowmodels/nwp_archives/out",
        logs = "/mnt/c/Users/horto/Data/snowmodels/nwp_archives/logs",
        fn = "/mnt/g/My Drive/snowmodels/ops/nwp_archives/gribts_functions.R"
    ),
    hpc = list(
        wd = "/scratch/shorton/nwp_archives",
        wgrib = "/home/shorton/sarpnwptools/wgrib2025/wgrib2/wgrib2",
        grib = paste0("/home/shorton/project/data/pre2025/grib/hrdps", ifelse(season < 2018, "_west", ""), "/", season),
        out = "/scratch/shorton/nwp_archives/out",
        logs = "/scratch/shorton/nwp_archives/logs",
        fn = "/scratch/shorton/nwp_archives/grib.R"
    )
)

# Select and initialize configuration based on environment
config <- base_paths[[if (test_mode) "test" else "hpc"]]
config$log_dir <- file.path(config$logs, season)
dir.create(config$log_dir, recursive = TRUE, showWarnings = FALSE)
setwd(config$wd)
source(config$fn)

# ===============================================================================
# 3. Generate Processing Schedule
# ===============================================================================

# Generate all days for the season (Sept 1 to May 31)
all_days <- as.character(seq(
    as.Date(paste0(season - 1, "-09-01")),
    as.Date(paste0(season, "-06-30")),
    by = "day"
))

if (!test_mode) {
    # HPC mode: Setup array job processing
    array_index <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID", "1"))
    total_jobs <- 8 # Total number of array jobs for parallel processing

    # Calculate days to process for this array job
    days_per_job <- ceiling(length(all_days) / total_jobs)
    start_idx <- (array_index - 1) * days_per_job + 1
    end_idx <- min(array_index * days_per_job, length(all_days))
    days <- all_days[start_idx:end_idx]

    # Log processing schedule
    cat(sprintf(
        "Processing days %d to %d (total: %d days)\n",
        start_idx, end_idx, length(days)
    ))
} else {
    # Test mode: Process all days sequentially
    days <- all_days
    cat(sprintf(
        "Test mode: Processing all %d days sequentially\n",
        length(days)
    ))
}

# ===============================================================================
# 4. Process Days
# ===============================================================================

start_time <- Sys.time()

# Process each day in sequence
for (day in days) {
    # Set up logging for this day
    log_file <- file.path(config$log_dir, paste0(gsub("-", "", day), ".log"))
    cat(sprintf("Starting day %s\n", day), file = log_file, append = TRUE)

    # Process the day
    prepare_day(day, log_file)

    # Log completion
    cat(sprintf("Completed day %s\n", day), file = log_file, append = TRUE)
}

# Log total processing time
duration <- difftime(Sys.time(), start_time, units = "hours")
if (!test_mode) {
    cat(sprintf("Completed chunk %d in %.2f hours\n", array_index, duration))
} else {
    cat(sprintf("Completed all days in %.2f hours\n", duration))
}

# ===============================================================================
# 5. Consolidate Log Files
# ===============================================================================

# Find all log files for this chunk/run
log_files <- list.files(
    config$log_dir,
    pattern = paste0("^(", paste(gsub("-", "", days), collapse = "|"), ")\\.log$"),
    full.names = TRUE
)

# Create consolidated log file
if (!test_mode) {
    output_log_file <- file.path(config$logs, paste0("season_", season, "_chunk_", array_index, ".log"))
} else {
    output_log_file <- file.path(config$logs, paste0("season_", season, "_test_run.log"))
}
file.remove(output_log_file)
for (log_file in log_files) {
    file.append(output_log_file, log_file)
}
