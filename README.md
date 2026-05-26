# splatx

`splatx` 是一个基于 **WebGPU (wgpu)** 的试验性 **3D & 4D Gaussian Splatting (4DGS)** 渲染器实现。

> [!CAUTION]
> **试验性项目**：本项目目前处于早期研发阶段，尚未进行深度性能优化。在当前实现下，复杂场景的渲染帧率低，主要用于算法验证与原型展示。

---

## 1. 技术实现原理

### 1.1 数据预处理与加载 (`src/model.rs`)
系统通过 CPU 预先解析包含几何参数与轻量级 MLP 权重的 `.npz` 模型文件。
- **协方差构建**：利用旋转四元数 $q$ 和缩放向量 $s$ 在上传前计算 3D 协方差矩阵 $\Sigma = RSS^T R^T$。
- **数据量化**：为了权衡显存占用，大部分参数采用 **FP16** 格式存储，并通过连续结构体数组（Storage Buffer）上传至 GPU。

### 1.2 渲染管线阶段 (`src/renderer/`)
渲染逻辑完全运行在 GPU 计算着色器（Compute Shader）中，分为以下步骤：
- **双重剔除 (`cull.rs`)**：
    - **时间过滤**：应用时间窗模型 $exp(-0.5 \cdot (\frac{t-t_0}{d})^2)$ 计算瞬时透明度。
    - **空间视锥剔除**：将高斯中心投影至剪裁空间，判断是否在视口范围内。
- **数据压缩 (`prefix_sum.rs` & `compact.rs`)**：
    使用并行前缀和算法统计可见点，并将稀疏的索引压缩为连续的“活跃索引”数组，以减少后续着色器的空转。
- **神经解码 (`appearance.rs`)**：
    在 GPU 上运行 OMG4 架构的 MLP 网络。通过 16 阶频率编码处理瞬时坐标，解码出每个基元的漫反射颜色（DC）、视角相关颜色补偿以及不透明度。
- **投影与排序 (`project.rs` & `sort.rs`)**：
    - 使用雅可比矩阵将 3D 高斯投影为 2D 椭圆。
    - 采用 GPU 基数排序（Radix Sort）对可见点按深度进行全局排序。
- **实例化光栅化 (`render.rs`)**：
    利用硬件实例化技术将高斯点展开为四边形，在片元着色器中进行高斯衰减计算并进行 Alpha 混合。

---

## 2. 模块结构

### 2.1 命令行工具 (`src/bin/`)
除了 Web 端支持，项目还包含了几个原生 Rust 程序：
- **`preview`**: 简单的桌面端预览程序，用于在开发阶段快速验证渲染效果。
- **`render-image`**: 离线渲染脚本，可将特定时间点的场景渲染并保存为 PNG 图片。
- **`inspect-npz`**: 辅助工具，用于打印和分析模型文件的张量统计信息。

### 2.2 TypeScript 库接口 (`ts/`)
本项目的前端核心接口层，负责与 WASM 模块通信。
- **`index.ts`**: 暴露给外部调用的 `Renderer` 类，管理渲染循环、相机控制及资源调度。
- **`worker.ts`**: 基于 Web Worker 的异步驱动，确保 WebGPU 重负载不影响浏览器主线程响应。

### 2.3 Web 演示应用 (`demo/`)
基于 `ts/` 库构建的交互式 Web 示例。
- 提供模型加载界面与交互式相机控制。

### 2.4 离线转换工具 (`tool/`)
- 包含 Python 脚本，用于将 OMG-4D/FTGS 原始权重转换为本项目兼容的 `.npz` 格式。

---

## 3. 快速开始

### 环境要求
- **Rust**: 1.80+
- **Bun**: 前端构建工具
- **WebGPU 兼容的浏览器** (如 Chrome 113+)

### 运行
1. **编译 WASM**: `bun run build:wasm`
2. **启动 Web Demo**: `bun run dev`
3. **原生预览**: `cargo run --release --bin preview -- <model_path>`
4. **离线渲染示例**: `cargo run --release --bin render-image -- demo/public/model/coffee_martini_S.npz`

---

## 4. Rust 接口使用示例

作为原生库引入时，可以通过以下方式调用 `splatx` 的渲染接口：

```rust
use glam::Vec3;
use splatx::{
    camera::Camera,
    model::SplatxModel,
    renderer::{RenderTarget, Renderer},
};

fn render_example(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output_view: &wgpu::TextureView,
    width: u32,
    height: u32,
) {
    // 1. 加载模型数据
    let model = SplatxModel::load_npz("path/to/model.npz").unwrap();

    // 2. 初始化渲染器
    let mut renderer = Renderer::new(device, queue, model);

    // 3. 设置相机参数
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, -3.0),
        target: Vec3::ZERO,
        up: Vec3::Y,
        fovy_radians: std::f32::consts::FRAC_PI_4,
        znear: 0.1,
        zfar: 100.0,
    };

    // 4. 创建 CommandEncoder 并配置 RenderTarget
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let target = RenderTarget {
        encoder: &mut encoder,
        queue,
        color_view: output_view,
        format: wgpu::TextureFormat::Rgba8Unorm, // 根据实际输出纹理格式调整
        width,
        height,
    };

    // 5. 执行渲染 (传入相机和当前时间)
    let time = 0.0;
    renderer.render(&camera, time, target);

    // 6. 提交指令到 GPU 队列
    queue.submit(Some(encoder.finish()));
}
```

---

## 5. 许可证

[MIT License](LICENSE)
