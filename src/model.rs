use std::{
    fs::File,
    io::{self, Read, Seek},
    path::Path,
};

use half::f16;
use ndarray::ArrayD;
use ndarray_npy::{NpzReader, ReadDataError, ReadableElement};
use py_literal::Value as PyValue;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const MLP_CONT_LEN: usize = 7168;
const MLP_DC_LEN: usize = 2048;
const MLP_SH_LEN: usize = 10240;
const MLP_OPACITY_LEN: usize = 2048;

#[derive(Debug)]
pub struct SplatxModel {
    pub means: Vec<[f16; 3]>,
    pub times: Vec<f16>,
    pub scales: Vec<[f16; 3]>,
    pub quats: Vec<[f16; 4]>,
    pub durations: Vec<f16>,
    pub velocities: Vec<[f16; 3]>,
    pub features_static: Vec<[f16; 3]>,
    pub features_view: Vec<[f16; 3]>,
    pub mlp_cont: Vec<f16>,
    pub mlp_dc: Vec<f16>,
    pub mlp_sh: Vec<f16>,
    pub mlp_opacity: Vec<f16>,
}

impl SplatxModel {
    pub fn load_npz(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        Self::load_npz_reader(file)
    }

    pub fn load_npz_reader(reader: impl Read + Seek) -> Result<Self> {
        let mut npz = NpzReader::new(reader)?;
        let len = gaussian_count(&mut npz)?;

        Ok(Self {
            means: read_rows(&mut npz, "means", len)?,
            times: read_scalars(&mut npz, "times", len)?,
            scales: read_rows(&mut npz, "scales", len)?,
            quats: read_rows(&mut npz, "quats", len)?,
            durations: read_scalars(&mut npz, "durations", len)?,
            velocities: read_rows(&mut npz, "velocities", len)?,
            features_static: read_rows(&mut npz, "features_static", len)?,
            features_view: read_rows(&mut npz, "features_view", len)?,
            mlp_cont: read_flat(&mut npz, "mlp_cont", MLP_CONT_LEN)?,
            mlp_dc: read_flat(&mut npz, "mlp_dc", MLP_DC_LEN)?,
            mlp_sh: read_flat(&mut npz, "mlp_sh", MLP_SH_LEN)?,
            mlp_opacity: read_flat(&mut npz, "mlp_opacity", MLP_OPACITY_LEN)?,
        })
    }

    pub fn len(&self) -> usize {
        self.means.len()
    }

    pub fn is_empty(&self) -> bool {
        self.means.is_empty()
    }
}

fn gaussian_count<R: Read + io::Seek>(npz: &mut NpzReader<R>) -> Result<usize> {
    let array = read_array(npz, "means")?;
    let shape = array.shape();
    let len = match shape {
        [rows, 3] => *rows,
        _ => return Err(format!("means must have shape [rows, 3], got {shape:?}").into()),
    };

    if len == 0 {
        return Err("model has no gaussians".into());
    }
    if len > u32::MAX as usize {
        return Err("model has too many gaussians".into());
    }

    Ok(len)
}

fn read_rows<R: Read + io::Seek, const N: usize>(
    npz: &mut NpzReader<R>,
    name: &str,
    rows: usize,
) -> Result<Vec<[f16; N]>> {
    let array = read_array(npz, name)?;
    let shape = array.shape();
    if shape != [rows, N] {
        return Err(format!("{name} must have shape [{rows}, {N}], got {shape:?}").into());
    }
    let slice = contiguous_slice(&array, name)?;

    Ok((0..rows)
        .map(|row| {
            let offset = row * N;
            std::array::from_fn(|column| slice[offset + column].0)
        })
        .collect())
}

fn read_scalars<R: Read + io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
    len: usize,
) -> Result<Vec<f16>> {
    let array = read_array(npz, name)?;
    let shape = array.shape();
    match shape {
        [n] if *n == len => Ok(array.iter().map(|value| value.0).collect()),
        [n, 1] if *n == len => Ok(array.iter().map(|value| value.0).collect()),
        _ => Err(format!("{name} must have shape [{len}] or [{len}, 1], got {shape:?}").into()),
    }
}

fn read_flat<R: Read + io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
    len: usize,
) -> Result<Vec<f16>> {
    let array = read_array(npz, name)?;
    let shape = array.shape();
    match shape {
        [n] if *n == len => Ok(array.iter().map(|value| value.0).collect()),
        _ => Err(format!("{name} must have shape [{len}], got {shape:?}").into()),
    }
}

fn read_array<R: Read + io::Seek>(npz: &mut NpzReader<R>, name: &str) -> Result<ArrayD<NpyF16>> {
    npz.by_name(name)
        .map_err(|err| format!("failed to read required npz array {name:?}: {err}").into())
}

fn contiguous_slice<'a>(array: &'a ArrayD<NpyF16>, name: &str) -> Result<&'a [NpyF16]> {
    array
        .as_slice_memory_order()
        .ok_or_else(|| format!("{name} is not contiguous").into())
}

#[derive(Clone, Copy, Debug)]
struct NpyF16(f16);

impl ReadableElement for NpyF16 {
    fn read_to_end_exact_vec<R: Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> std::result::Result<Vec<Self>, ReadDataError> {
        let endian = match type_desc {
            PyValue::String(desc) if desc == "<f2" => Endian::Little,
            PyValue::String(desc) if desc == ">f2" => Endian::Big,
            PyValue::String(desc) if desc == "|f2" => Endian::Native,
            other => return Err(ReadDataError::WrongDescriptor(other.clone())),
        };

        let byte_len = len.checked_mul(2).ok_or_else(|| {
            ReadDataError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "f16 byte length overflow",
            ))
        })?;
        let mut bytes = vec![0_u8; byte_len];
        reader.read_exact(&mut bytes)?;

        let mut extra = [0_u8; 1];
        if reader.read(&mut extra)? != 0 {
            return Err(ReadDataError::ExtraBytes(1));
        }

        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = match endian {
                    Endian::Little => u16::from_le_bytes([chunk[0], chunk[1]]),
                    Endian::Big => u16::from_be_bytes([chunk[0], chunk[1]]),
                    Endian::Native => u16::from_ne_bytes([chunk[0], chunk[1]]),
                };
                NpyF16(f16::from_bits(bits))
            })
            .collect())
    }
}

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
    Native,
}

mod display {
    use super::{MLP_CONT_LEN, MLP_DC_LEN, MLP_OPACITY_LEN, MLP_SH_LEN, SplatxModel};
    use half::f16;
    use std::fmt;

    impl fmt::Display for SplatxModel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(f, "gaussians: {}", self.len())?;
            write_vec_array(f, "means", &[self.len(), 3], &self.means)?;
            write_slice(f, "times", &[self.len()], &self.times)?;
            write_vec_array(f, "scales", &[self.len(), 3], &self.scales)?;
            write_vec_array(f, "quats", &[self.len(), 4], &self.quats)?;
            write_slice(f, "durations", &[self.len()], &self.durations)?;
            write_vec_array(f, "velocities", &[self.len(), 3], &self.velocities)?;
            write_vec_array(
                f,
                "features_static",
                &[self.len(), 3],
                &self.features_static,
            )?;
            write_vec_array(f, "features_view", &[self.len(), 3], &self.features_view)?;
            write_slice(f, "mlp_cont", &[MLP_CONT_LEN], &self.mlp_cont)?;
            write_slice(f, "mlp_dc", &[MLP_DC_LEN], &self.mlp_dc)?;
            write_slice(f, "mlp_sh", &[MLP_SH_LEN], &self.mlp_sh)?;
            write_slice(f, "mlp_opacity", &[MLP_OPACITY_LEN], &self.mlp_opacity)
        }
    }

    fn write_slice(
        f: &mut fmt::Formatter<'_>,
        name: &str,
        shape: &[usize],
        values: &[f16],
    ) -> fmt::Result {
        write_summary(
            f,
            name,
            shape,
            preview_slice(values),
            stats(values.iter().copied()),
        )
    }

    fn write_vec_array<const N: usize>(
        f: &mut fmt::Formatter<'_>,
        name: &str,
        shape: &[usize],
        values: &[[f16; N]],
    ) -> fmt::Result {
        write_summary(
            f,
            name,
            shape,
            preview_vec_array(values),
            stats(values.iter().flatten().copied()),
        )
    }

    fn write_summary(
        f: &mut fmt::Formatter<'_>,
        name: &str,
        shape: &[usize],
        preview: String,
        stats: Stats,
    ) -> fmt::Result {
        writeln!(
            f,
            "{name}: shape={shape:?} dtype=f16 len={} finite={} mean={:.9} var={:.9} min={:.9} max={:.9} first3={preview}",
            stats.len, stats.finite, stats.mean, stats.variance, stats.min, stats.max,
        )
    }

    fn preview_slice(values: &[f16]) -> String {
        let items = values
            .iter()
            .take(3)
            .map(|value| format!("{:.9}", value.to_f32()))
            .collect::<Vec<_>>();
        format!("[{}]", items.join(", "))
    }

    fn preview_vec_array<const N: usize>(values: &[[f16; N]]) -> String {
        let rows = values
            .iter()
            .take(3)
            .map(|row| {
                let items = row
                    .iter()
                    .map(|value| format!("{:.9}", value.to_f32()))
                    .collect::<Vec<_>>();
                format!("[{}]", items.join(", "))
            })
            .collect::<Vec<_>>();
        format!("[{}]", rows.join(", "))
    }

    fn stats(values: impl Iterator<Item = f16>) -> Stats {
        let mut len = 0_usize;
        let mut finite = 0_usize;
        let mut mean = 0_f64;
        let mut m2 = 0_f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for value in values.map(|value| value.to_f32()) {
            len += 1;
            if !value.is_finite() {
                continue;
            }

            finite += 1;
            min = min.min(value);
            max = max.max(value);

            let value = value as f64;
            let delta = value - mean;
            mean += delta / finite as f64;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }

        let variance = if finite > 0 {
            m2 / finite as f64
        } else {
            f64::NAN
        };
        if finite == 0 {
            min = f32::NAN;
            max = f32::NAN;
        }

        Stats {
            len,
            finite,
            mean,
            variance,
            min,
            max,
        }
    }

    struct Stats {
        len: usize,
        finite: usize,
        mean: f64,
        variance: f64,
        min: f32,
        max: f32,
    }
}
