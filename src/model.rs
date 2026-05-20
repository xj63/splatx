use std::{
    fs::File,
    io::{self, Read, Seek},
    path::Path,
};

use half::f16;
use ndarray::ArrayD;
use ndarray_npy::{NpzReader, ReadDataError, ReadableElement};
use py_literal::Value as PyValue;

pub type F16Array = ArrayD<f16>;

#[derive(Debug)]
pub struct SplatxModel {
    pub means: F16Array,
    pub times: F16Array,
    pub scales: F16Array,
    pub quats: F16Array,
    pub durations: F16Array,
    pub velocities: F16Array,
    pub features_static: F16Array,
    pub features_view: F16Array,
    pub mlp_cont: F16Array,
    pub mlp_dc: F16Array,
    pub mlp_sh: F16Array,
    pub mlp_opacity: F16Array,
}

impl SplatxModel {
    pub fn load_npz(
        path: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path)?;
        Self::load_npz_reader(file)
    }

    pub fn load_npz_reader(
        reader: impl Read + Seek,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut npz = NpzReader::new(reader)?;
        let names = npz.names()?.into_iter().collect::<Vec<_>>();

        let means = read_required_f16(&mut npz, &names, "means")?;
        let times = read_required_f16(&mut npz, &names, "times")?;
        let scales = read_required_f16(&mut npz, &names, "scales")?;
        let quats = read_required_f16(&mut npz, &names, "quats")?;
        let durations = read_required_f16(&mut npz, &names, "durations")?;
        let velocities = read_required_f16(&mut npz, &names, "velocities")?;
        let features_static = read_required_f16(&mut npz, &names, "features_static")?;
        let features_view = read_required_f16(&mut npz, &names, "features_view")?;
        let mlp_cont = read_required_f16(&mut npz, &names, "mlp_cont")?;
        let mlp_dc = read_required_f16(&mut npz, &names, "mlp_dc")?;
        let mlp_sh = read_required_f16(&mut npz, &names, "mlp_sh")?;
        let mlp_opacity = read_required_f16(&mut npz, &names, "mlp_opacity")?;

        Ok(Self {
            means,
            times,
            scales,
            quats,
            durations,
            velocities,
            features_static,
            features_view,
            mlp_cont,
            mlp_dc,
            mlp_sh,
            mlp_opacity,
        })
    }

    pub fn len(&self) -> usize {
        self.means.shape().first().copied().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn load_npz(
    path: impl AsRef<Path>,
) -> Result<SplatxModel, Box<dyn std::error::Error + Send + Sync>> {
    SplatxModel::load_npz(path)
}

fn read_required_f16<R: Read + io::Seek>(
    npz: &mut NpzReader<R>,
    names: &Vec<String>,
    name: &str,
) -> Result<F16Array, Box<dyn std::error::Error + Send + Sync>> {
    if !names.contains(&name.to_string()) {
        return Err(format!("missing required npz array {name:?}").into());
    }

    read_f16(npz, name)
}

fn read_f16<R: Read + io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<F16Array, Box<dyn std::error::Error + Send + Sync>> {
    let raw: ArrayD<NpyF16> = npz.by_name(name)?;
    Ok(raw.mapv(|value| value.0))
}

#[derive(Clone, Copy, Debug)]
struct NpyF16(f16);

impl ReadableElement for NpyF16 {
    fn read_to_end_exact_vec<R: Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError> {
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
