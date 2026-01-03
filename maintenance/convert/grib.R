# ===============================================================================
# GRIB Processing Functions
# Author: Simon Horton
# Description: Core functions for processing GRIB files in the NWP archives pipeline
# ===============================================================================

# ===============================================================================
# 1. Inventory and Data Extraction Functions
# ===============================================================================

#' Extract inventory information from a GRIB file
#' @param fname Path to the GRIB file
#' @return Data frame containing inventory information
inventory_gribfile <- function(fname) {
    out <- system2(config$wgrib, args = c(fname, "-s"), stdout = TRUE)
    df <- do.call(rbind, strsplit(out, ":"))
    data.frame(
        id = as.numeric(df[, 1]),
        run = as.POSIXct(sub("^d=([0-9]+)$", "\\1", df[, 3]), format = "%Y%m%d%H", tz = "UTC"),
        var = df[, 4],
        level = df[, 5],
        phour = sapply(df[, 6], convert_to_hours),
        fname = fname,
        timestamp = as.POSIXct(sub("^d=([0-9]+)$", "\\1", df[, 3]), format = "%Y%m%d%H", tz = "UTC") +
            sapply(df[, 6], convert_to_hours) * 3600
    )
}

#' Convert forecast time strings to hours
#' @param x Forecast time string
#' @return Numeric value in hours
convert_to_hours <- function(x) {
    if (x == "anl") {
        return(0)
    }

    if (grepl("min fcst$", x)) {
        return(as.numeric(sub("^([0-9]+) min fcst$", "\\1", x)) / 60)
    }
    if (grepl("hour acc fcst$", x)) {
        return(as.numeric(sub("^0-([0-9]+) hour acc fcst$", "\\1", x)))
    }
    if (grepl("day acc fcst$", x)) {
        return(as.numeric(sub("^0-([0-9]+) day acc fcst$", "\\1", x)) * 24)
    }

    warning(paste("Unknown forecast time format:", x))
    NA
}

#' Get inventory for a specific day
#' @param day Date string in YYYY-MM-DD format
#' @return Data frame containing inventory for the specified day
get_inventory <- function(day) {
    dates <- c(day, format(as.Date(day) - 1, "%Y-%m-%d"))
    files <- unlist(lapply(dates, function(d) {
        list.files(config$grib, pattern = gsub("-", "", d), full.names = TRUE)
    }))
    inv <- do.call(rbind, lapply(files, inventory_gribfile))
    inv[inv$level %in% c("10 m above ground", "2 m above ground", "surface") & inv$var != "PRES", ]
}

#' Select best records for each hour and variable
#' @param inv Inventory data frame
#' @param day Date string in YYYY-MM-DD format
#' @return Data frame with selected records
select_best_records <- function(inv, day) {
    hrs <- seq(as.POSIXct(day, format = "%Y-%m-%d", tz = "UTC"), by = "hour", length.out = 24)
    selected <- do.call(rbind, lapply(hrs, function(hr) {
        subset <- inv[inv$timestamp == hr, ]
        do.call(rbind, lapply(unique(subset$var), function(var) {
            var_records <- subset[subset$var == var, ]
            priority <- var_records[var_records$phour >= 7 & var_records$phour <= 12, ]
            if (nrow(priority) > 0) {
                priority[order(priority$phour), ][1, ]
            } else {
                var_records[order(var_records$phour), ][1, ]
            }
        }))
    }))

    # Categorize variables
    var_types <- list(
        A = c("TMP", "WIND", "WDIR", "PRATE"),
        B = c("RH", "DEPR"),
        C = c("DSWRF", "DLWRF", "APCP"),
        D = c("RPRATE", "SPRATE")
    )
    selected$type <- sapply(selected$var, function(v) {
        names(var_types)[sapply(var_types, function(x) v %in% x)]
    })
    selected[order(selected$var, selected$timestamp), ]
}

# ===============================================================================
# 2. GRIB Processing Functions
# ===============================================================================

#' Write selected records to output GRIB file
#' @param records Data frame of records to write
#' @param output_file Path to output GRIB file
write_records <- function(records, output_file) {
    rle_files <- rle(records$fname)
    start_idx <- cumsum(c(1, rle_files$lengths[-length(rle_files$lengths)]))
    for (i in seq_along(rle_files$values)) {
        idx <- start_idx[i]:(start_idx[i] + rle_files$lengths[i] - 1)
        group_records <- records[idx, ]
        system2("bash", args = c("-c", shQuote(paste0(
            config$wgrib, " ", rle_files$values[i],
            " | egrep '(", paste0("^", group_records$id, ":", collapse = "|"), ")' | ",
            config$wgrib, " -i ", rle_files$values[i], " -append -set_grib_type complex3 -grib_out ", output_file
        ))), stdout = FALSE)
    }
}

#' Process humidity data from dew point depression
#' @param depr_records Records containing dew point depression data
#' @param inv_raw Raw inventory data
#' @param output_file Path to output GRIB file
#' @param temp_dir Directory for temporary files
process_humidity <- function(depr_records, inv_raw, output_file, temp_dir) {
    temp_files <- character(0)

    tryCatch({
        for (i in seq_len(nrow(depr_records))) {
            depr <- depr_records[i, ]
            tmp <- inv_raw[inv_raw$timestamp == depr$timestamp &
                inv_raw$run == depr$run &
                inv_raw$var == "TMP", ]

            temp_file <- file.path(temp_dir, paste0("temp_rh_", i, ".grb"))
            temp_files <- c(temp_files, temp_file)

            # Create the temporary file
            system2("bash", args = c("-c", shQuote(paste0(
                config$wgrib, " ", tmp$fname,
                " | egrep '(", paste0("^", tmp$id, ":", "|^", depr$id, ":"), ")' | ",
                config$wgrib, " -i ", tmp$fname,
                " -if ':TMP:' -rpn sto_1 -fi",
                " -if ':DEPR:' -rpn sto_2 -fi",
                " -if_reg 1:2",
                " -rpn 'rcl_1:273.15:-:sto_3'",
                " -rpn 'rcl_1:rcl_2:-:273.15:-:sto_4'",
                " -rpn 'rcl_3:17.62:*:rcl_3:243.12:+:/:exp:6.112:*:sto_5'",
                " -rpn 'rcl_4:17.62:*:rcl_4:243.12:+:/:exp:6.112:*:sto_6'",
                " -rpn 'rcl_6:rcl_5:/:100:*'",
                " -set_var RH -set_grib_type complex3 -grib_out ", temp_file
            ))), stdout = FALSE)

            # Wait for file to be created and readable
            Sys.sleep(0.1) # Small delay to ensure file system sync
            if (!file.exists(temp_file)) {
                stop("Failed to create temporary file: ", temp_file)
            }

            # Append to output file
            system2(config$wgrib, args = c(temp_file, "-append", "-grib", output_file), stdout = FALSE)
        }
    }, finally = {
        # Clean up only after all processing is complete
        unlink(temp_files)
    })
}


#########################################
## process_prate
# "$WGRIB" "$grib_file" | grep ":PRATE:" | "$WGRIB" -i "$grib_file" \
#     -if ':PRATE:' -rpn '3600:*' \
#     -set_var APCP \
#     -set_grib_type complex3 \
#     -grib_out "$output_file"
#########################################


#' Process accumulated variables
#' @param var_records Records containing accumulated variables
#' @param inv_raw Raw inventory data
#' @param output_file Path to output GRIB file
#' @param temp_dir Directory for temporary files
process_accumulated <- function(var_records, inv_raw, output_file, temp_dir) {
    temp_files <- character(0)
    on.exit(unlink(temp_files))

    for (var in c("DSWRF", "DLWRF", "APCP")) {
        records <- var_records[var_records$var == var, ]
        if (nrow(records) == 0) next

        for (run in split(records[!names(records) %in% "type"], records$run)) {
            min_phour <- min(run$phour)
            if (min_phour > 1) {
                prev_record <- inv_raw[inv_raw$var == var &
                    inv_raw$run == run$run[1] &
                    inv_raw$phour == (min_phour - 1), ]
                run <- rbind(prev_record, run)
            }
            run <- run[order(run$phour), ]

            temp_file <- file.path(temp_dir, paste0("temp_", tolower(var), "_", run$id[1], ".grb"))
            temp_files <- c(temp_files, temp_file)

            cmd <- paste0(
                config$wgrib, " ", run$fname[1],
                " | egrep '(", paste0("^", run$id, ":", collapse = "|"), ")' | ",
                config$wgrib, " -i ", run$fname[1]
            )
            if (var != "APCP") cmd <- paste0(cmd, " -if ':", var, ":' -rpn '0.0002778:*'")
            cmd <- paste0(cmd, " -set_grib_type complex3 -ncep_norm ", temp_file)
            system2("bash", args = c("-c", shQuote(cmd)), stdout = FALSE)

            if (min(run$phour) < min_phour) {
                system2(config$wgrib, args = c(
                    temp_file,
                    "-match", paste0("'(", paste0(paste0("^", which(run$phour >= min_phour)), ":", collapse = "|"), ")'"),
                    "-append", "-grib", output_file
                ), stdout = FALSE)
            } else {
                system2(config$wgrib, args = c(temp_file, "-append", "-grib", output_file), stdout = FALSE)
            }
        }
    }
}

#' Reformat GRIB file with consistent time steps
#' @param output_file Path to output GRIB file
#' @param temp_dir Directory for temporary files
reformat_grib <- function(output_file, temp_dir) {
    temp_files <- file.path(temp_dir, paste0("temp", 1:2, "_", basename(output_file)))
    on.exit(unlink(temp_files))

    system2(config$wgrib, args = c(output_file, "-set_lev", "surface", "-grib", temp_files[1]), stdout = FALSE)
    inv <- system2(config$wgrib, args = c(temp_files[1], "-vt"), stdout = TRUE)

    min_vt <- min(na.omit(sapply(inv, function(x) {
        vt <- sub(".*vt=([0-9]{10}).*", "\\1", x)
        if (nchar(vt) == 10) vt else NA
    })))

    for (i in seq_along(inv)) {
        vt <- sub(".*vt=([0-9]{10}).*", "\\1", inv[i])
        if (nchar(vt) != 10) next

        hours <- as.numeric(difftime(
            as.POSIXct(vt, format = "%Y%m%d%H", tz = "UTC"),
            as.POSIXct(min_vt, format = "%Y%m%d%H", tz = "UTC"),
            units = "hours"
        ))

        system2("bash", args = c("-c", shQuote(paste0(
            config$wgrib, " ", temp_files[1],
            " -d ", i,
            " -set_date ", min_vt,
            " -set_ftime '", hours, " hour fcst'",
            " -append -grib ", temp_files[2]
        ))), stdout = FALSE)
    }

    file.rename(temp_files[2], output_file)
}

#' Regrid GRIB file to 2023 specifications
#' @param output_file Path to output GRIB file
#' @param temp_dir Directory for temporary files
regrid2023 <- function(output_file, temp_dir) {
    # Grid specifications for 2023
    grid_spec <- "nps:252.000000:60.000000 231.917464:2576:2500.000000 35.603376:1456:2500.000000"
    temp_file <- file.path(temp_dir, paste0("temp_", basename(output_file)))
    system2(config$wgrib, args = c(output_file, "-new_grid_winds", "grid", "-new_grid", grid_spec, temp_file), stdout = FALSE)
    file.rename(temp_file, output_file)
}

#' Process precipitation rate grids by converting to hourly accumulation
#' @param var_records Records containing PRATE data
#' @param output_file Path to output GRIB file
#' @param temp_dir Directory for temporary files
process_prate <- function(var_records, output_file, temp_dir) {
    temp_files <- character(0)
    on.exit(unlink(temp_files))

    records <- var_records[var_records$var == "PRATE", ]
    if (nrow(records) == 0) {
        return()
    }

    for (run in split(records[!names(records) %in% "type"], records$run)) {
        run <- run[order(run$phour), ]

        temp_file <- file.path(temp_dir, paste0("temp_prate_", run$id[1], ".grb"))
        temp_files <- c(temp_files, temp_file)

        # Extract PRATE records and multiply by 3600 to convert to hourly accumulation
        cmd <- paste0(
            config$wgrib, " ", run$fname[1],
            " | egrep '(", paste0("^", run$id, ":", collapse = "|"), ")' | ",
            config$wgrib, " -i ", run$fname[1],
            " -if ':PRATE:' -rpn '3600:*'",
            " -set_var APCP", # Rename to APCP to indicate it's now accumulation
            " -set_grib_type complex3 -ncep_norm ", temp_file
        )
        system2("bash", args = c("-c", shQuote(cmd)), stdout = FALSE)

        # Append to output file
        system2(config$wgrib, args = c(temp_file, "-append", "-grib", output_file), stdout = FALSE)
    }
}

# ===============================================================================
# 3. Main Processing Function
# ===============================================================================

#' Process a single day of GRIB data
#' @param day Date string in YYYY-MM-DD format
#' @param log_file Path to log file
prepare_day <- function(day, log_file) {
    start_time <- Sys.time()
    cat("Starting processing for", day, "at", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n", file = log_file, append = TRUE)

    tryCatch(
        {
            # Create unique temp directory for this process
            temp_dir <- file.path(config$out, paste0("temp_", gsub("-", "", day)))
            dir.create(temp_dir, showWarnings = FALSE)
            on.exit(unlink(temp_dir, recursive = TRUE))

            inv_raw <- get_inventory(day)
            if (is.null(inv_raw) || nrow(inv_raw) == 0) {
                cat("No records found for", day, "\n", file = log_file, append = TRUE)
                return()
            }

            inv <- select_best_records(inv_raw, day)
            if (nrow(inv) == 0) {
                cat("No valid records found for", day, "\n", file = log_file, append = TRUE)
                return()
            }

            cat("Inventory done at", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n", file = log_file, append = TRUE)

            output_file <- file.path(config$out, paste0("HRDPS_", format(as.Date(day), "%Y%m%d"), ".grib2"))
            try(file.remove(output_file))

            # Process each data type
            vars <- unique(inv$var)
            data_types <- split(inv, inv$type)

            write_records(data_types$A, output_file)
            if ("RH" %in% vars) {
                write_records(data_types$B, output_file)
            } else if ("DEPR" %in% vars) {
                process_humidity(inv[inv$var == "DEPR", ], inv_raw, output_file, temp_dir)
            }
            cat("Basic variables done at", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n", file = log_file, append = TRUE)

            process_accumulated(data_types$C, inv_raw, output_file, temp_dir)
            cat("Accumulated variables done at", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n", file = log_file, append = TRUE)
            # if (any(c("RPRATE", "SPRATE") %in% vars)) {
            #     write_records(data_types$D, output_file)
            # }

            reformat_grib(output_file, temp_dir)

            # if (day >= "2023-02-23" && day < "2023-07-01") regrid2023(output_file, temp_dir)

            duration <- difftime(Sys.time(), start_time, units = "secs")
            cat("Completed", day, "at", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "in", round(duration, 2), "seconds\n", file = log_file, append = TRUE)
        },
        error = function(e) {
            cat("Error processing", day, ":", e$message, "\n", file = log_file, append = TRUE)
        }
    )
}
