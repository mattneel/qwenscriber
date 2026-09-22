//! WebGPU flag constants, written out.
//!
//! The WebGPU *interfaces* (`GPUDevice`, `GPUBuffer`, `GPUPipelineLayout`, ...) come from the DOM
//! lib, but the constant objects that carry the bit flags do not, and this package carries zero
//! production dependencies -- no `@webgpu/types` to take them from. So the numbers live here, once,
//! named as the specification names them, and every other module refers to these names instead of
//! to a literal. Only the flags this SDK actually sets are listed.

/** `GPUBufferUsage` (WebGPU specification, buffer creation). */
export const BUFFER_USAGE = {
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  MAP_READ: 0x0001,
} as const;

/** `GPUMapMode` (WebGPU specification, buffer mapping). */
export const MAP_MODE = {
  READ: 0x0001,
} as const;

/** `GPUShaderStage` (WebGPU specification, bind group layout visibility). */
export const SHADER_STAGE = {
  COMPUTE: 0x4,
} as const;
