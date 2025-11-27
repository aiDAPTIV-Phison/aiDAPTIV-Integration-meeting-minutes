#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use log;
use env_logger;
use std::io::Write;

fn main() {
    std::env::set_var("RUST_LOG", "info");

    // In release mode on Windows, write logs to file since there's no console
    #[cfg(all(not(debug_assertions), target_os = "windows"))]
    {
        use std::fs::OpenOptions;
        use env_logger::Builder;

        // Get log file path in user's AppData/Local/meetily
        let log_path = dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("meetily")
            .join("meetily.log");

        // Ensure directory exists
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // Open log file (append mode to preserve history)
        if let Ok(file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Builder::from_default_env()
                .format(|buf, record| {
                    writeln!(
                        buf,
                        "{} [{}] {} - {}",
                        chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                        record.level(),
                        record.target(),
                        record.args()
                    )
                })
                .target(env_logger::Target::Pipe(Box::new(file)))
                .init();

            log::info!("Log file initialized at: {:?}", log_path);
        } else {
            // Fallback to default if file creation fails
            env_logger::init();
        }
    }

    // Debug mode or non-Windows: use default console logging
    #[cfg(any(debug_assertions, not(target_os = "windows")))]
    {
        env_logger::init();
    }

    // Async logger will be initialized lazily when first needed (after Tauri runtime starts)
    log::info!("Starting application...");
    app_lib::run();
}
