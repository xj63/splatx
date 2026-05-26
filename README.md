# splatx

`splatx` 是一个基于 **WebGPU (wgpu)** 的高性能 **3D & 4D Gaussian Splatting (高斯泼溅)** 渲染器。它采用 Rust 编写核心逻辑，并通过 WebAssembly (WASM) 运行在浏览器中，利用 GPU 的计算能力实现流畅的动态场景渲染。

## 项目特点

- **高性能渲染**：完全基于 WebGPU 的计算管线，支持数百万个高斯点的实时渲染。
- **4D 动态支持**：支持随时间变化的动态场景（4DGS），内置 MLP 调制和时间相关的位置、颜色及不透明度计算。
- **混合架构**：核心渲染引擎由 Rust 编写，Web 端通过 TypeScript 与 WebWorker 进行异步编排，确保 UI 响应不受渲染压力影响。
- **完善的管线**：包含完整的显存管理、视锥剔除（Culling）、并行前缀和（Prefix Sum）、基数排序（Radix Sort）等 GPU 加速模块。
- **端到端工具链**：提供 Python 脚本，支持从常见 4DGS 格式（如 OMG-4D/FTGS）转换为项目专有的 `.npz` 格式。

---

## 模块介绍

项目代码结构清晰，分为核心算法、渲染管线、Web 绑定和辅助工具四个部分：

### 1. 核心模型与数据 (`src/model.rs`)
负责 3D/4D 高斯点数据的加载与存储。
- **数据结构**：包含位置（Means）、时间（Times）、缩放（Scales）、旋转（Quats）、持续时间（Durations）、速度（Velocities）以及用于动态外观调制的 MLP 权重。
- **加载器**：支持从压缩的 `.npz` 文件中高效读取半精度浮点数（f16）数据。

### 2. 渲染引擎 (`src/renderer/`)
这是项目的核心，实现了一个多阶段的 GPU 渲染管线：
- **CullStage (`cull.rs`)**：视锥剔除与时间过滤。根据当前相机视角和时间戳，在 GPU 上并行标记可见的高斯点。
- **PrefixSumStage (`prefix_sum.rs`)**：实现并行前缀和算法。利用 GPU Subgroup 特性，快速计算可见点的偏移量，为后续的压缩（Compact）做准备。
- **CompactStage (`compact.rs`)**：将可见点的索引压缩到连续的缓冲区中，极大减少后续阶段的计算量。
- **IndirectStage (`indirect.rs`)**：动态生成 GPU 间接绘制指令（Indirect Draw/Dispatch），使管线能根据剔除结果自动调整计算规模。
- **AppearanceStage (`appearance.rs`)**：计算高斯点的颜色和不透明度。包含 4D 动态逻辑：
    - 根据速度（Velocity）计算当前时间的位置偏移。
    - 使用 GPU 实现的轻量级 MLP 计算随时间变化的材质属性。
    - 计算时间衰减的不透明度（Temporal Opacity）。
- **ProjectStage (`project.rs`)**：将 3D/4D 高斯点投影到 2D 屏幕空间，计算其在屏幕上的位置、深度和 2D 协方差矩阵。
- **SortStage (`sort.rs`)**：基于深度的高性能基数排序（Radix Sort），确保渲染时的 Alpha 合成顺序正确。
- **RenderStage (`render.rs`)**：最终的着色阶段，根据排序后的结果执行 Tile-based 渲染或传统的 Alpha Blending。

### 3. Web 绑定与交互 (`ts/`, `src/web.rs`)
- **TypeScript 封装**：提供易用的 API，支持 `OffscreenCanvas` 渲染。
- **Web Worker**：渲染引擎运行在独立的 Worker 线程中，通过消息机制与主线程通信。
- **相机控制**：内置 Orbit 控制器，支持平滑的缩放、旋转和平移交互。

### 4. 转换工具 (`tool/`)
- **`convert-omg4ftgs-to-splatx.py`**：将 OMG-4D 等模型的权重和数据转换为本项目的高效二进制格式。
- **`poses_bounds_npy_to_json.py`**：处理相机路径和边界数据。

---

## 核心渲染管线流程

每一帧的渲染都经过以下严密的 GPU 计算步骤：

1.  **可见性测试**：在 Compute Shader 中并行检查每个高斯点是否在视锥内且处于存活时间内。
2.  **数据压缩**：通过 Prefix Sum 统计可见点总数，并将它们的索引重新排列，消除无效点。
3.  **动态调制**：执行 Appearance Shader，应用 MLP 权重和速度场，计算该时间点下高斯点的真实物理属性。
4.  **屏幕投影**：计算 3D 高斯体在 2D 图像上的投影形态。
5.  **深度排序**：对所有可见点按从远到近排序。
6.  **光栅化渲染**：在 FP16 浮点纹理上进行 Alpha 合成，最后通过 Blit 转换为标准颜色空间显示。

---

## 快速开始

### 环境依赖
- **Rust**: 安装最新稳定的 Rust 工具链。
- **Bun**: 用于前端构建和包管理。
- **wasm-pack**: 用于编译 Rust 为 WebAssembly。

### 构建与运行
1.  **编译 WASM 核心**：
    ```bash
    bun run build:wasm
    ```
2.  **启动开发服务器**：
    ```bash
    bun run dev
    ```
    启动后访问 `http://localhost:5173` 查看示例。

3.  **生产环境打包**：
    ```bash
    bun run build
    ```

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
