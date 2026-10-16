//! A design may carry `simulation`: named simulation studies (`crates/doc/src/simulation.rs`),
//! each with a cell size, a domain box, per-body heat and conductivity settings, wall, air,
//! thermal and fan settings and optional airflow settings. A design without studies saves as it
//! always did, so older designs need no transformation; 0.14.x asks for an update instead of
//! meeting a field it can't read. Results are never saved.
