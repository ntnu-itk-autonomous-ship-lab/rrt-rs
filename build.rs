//! build.rs
extern crate built;

fn main() {
    built::write_built_file().expect("Failed to acquire build-time information");
    pyo3_build_config::add_extension_module_link_args();
}
