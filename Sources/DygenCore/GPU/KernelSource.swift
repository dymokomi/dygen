import Foundation

/// All compute kernels as MSL source, compiled once at runtime into one library
/// (avoids Xcode 26's separate Metal-toolchain component). Grows across M5–M7.
enum KernelSource {
    static let all = """
    #include <metal_stdlib>
    using namespace metal;

    // Passthrough copy.
    kernel void copy_tex(texture2d<float, access::write> out [[texture(0)]],
                         texture2d<float, access::read> src [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        out.write(src.read(gid), gid);
    }

    // Block pixelation: each block samples its top-left source pixel.
    kernel void pixelate(texture2d<float, access::write> out [[texture(0)]],
                         texture2d<float, access::read> src [[texture(1)]],
                         constant uint& block [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        uint b = max(block, 1u);
        uint2 c = (gid / b) * b;
        out.write(src.read(c), gid);
    }

    // Nearest-palette index for each pixel.
    static inline uint nearest_palette(float3 c, constant float4* palette, uint count) {
        uint best = 0; float bestD = 1e20;
        for (uint i = 0; i < count; i++) {
            float3 d = c - palette[i].rgb;
            float dist = dot(d, d);
            if (dist < bestD) { bestD = dist; best = i; }
        }
        return best;
    }

    // Map each pixel to its nearest palette index (index texture only).
    kernel void assign_index(texture2d<uint, access::write> outIdx [[texture(0)]],
                             texture2d<float, access::read> src [[texture(1)]],
                             constant float4* palette [[buffer(0)]],
                             constant uint& count [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= outIdx.get_width() || gid.y >= outIdx.get_height()) return;
        uint best = nearest_palette(src.read(gid).rgb, palette, count);
        outIdx.write(uint4(best, 0, 0, 0), gid);
    }

    // Quantize: write both the palette colour and the index.
    kernel void quantize_assign(texture2d<float, access::write> outColor [[texture(0)]],
                                texture2d<uint, access::write> outIdx [[texture(1)]],
                                texture2d<float, access::read> src [[texture(2)]],
                                constant float4* palette [[buffer(0)]],
                                constant uint& count [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= outColor.get_width() || gid.y >= outColor.get_height()) return;
        uint best = nearest_palette(src.read(gid).rgb, palette, count);
        outColor.write(float4(palette[best].rgb, 1.0), gid);
        outIdx.write(uint4(best, 0, 0, 0), gid);
    }

    // Mode (most-common index) filter over a window — despeckles the blobs.
    kernel void cleanup_mode(texture2d<float, access::write> outColor [[texture(0)]],
                             texture2d<uint, access::write> outIdx [[texture(1)]],
                             texture2d<uint, access::read> idx [[texture(2)]],
                             constant float4* palette [[buffer(0)]],
                             constant uint& count [[buffer(1)]],
                             constant int& radius [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
        int W = int(outColor.get_width()), H = int(outColor.get_height());
        if (int(gid.x) >= W || int(gid.y) >= H) return;
        uint n = min(count, 64u);
        uint counts[64];
        for (uint i = 0; i < n; i++) counts[i] = 0;
        int r = radius;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                int x = clamp(int(gid.x) + dx, 0, W - 1);
                int y = clamp(int(gid.y) + dy, 0, H - 1);
                uint id = idx.read(uint2(x, y)).r;
                if (id < n) counts[id]++;
            }
        }
        uint best = 0, bestC = 0;
        for (uint i = 0; i < n; i++) { if (counts[i] > bestC) { bestC = counts[i]; best = i; } }
        outColor.write(float4(palette[best].rgb, 1.0), gid);
        outIdx.write(uint4(best, 0, 0, 0), gid);
    }

    // Block pixelation of an index texture (keeps masks aligned with blobs).
    kernel void pixelate_index(texture2d<uint, access::write> out [[texture(0)]],
                               texture2d<uint, access::read> src [[texture(1)]],
                               constant uint& block [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        uint b = max(block, 1u);
        out.write(src.read((gid / b) * b), gid);
    }

    // ---- Painterly (M7) ----

    static inline float hash1(uint n) {
        n = (n << 13) ^ n;
        n = n * (n * n * 15731u + 789221u) + 1376312589u;
        return float(n & 0x7fffffffu) / float(0x7fffffff);
    }

    // Pass 1: recolour each region with its palette colour + a per-colour jitter.
    kernel void paint_blobs(texture2d<float, access::write> out [[texture(0)]],
                            texture2d<uint, access::read> idx [[texture(1)]],
                            constant float4* palette [[buffer(0)]],
                            constant uint& count [[buffer(1)]],
                            constant uint& seed [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        uint i = min(idx.read(gid).r, count - 1);
        float j = (hash1(i * 2654435761u + seed) * 6.0 - 3.0) / 255.0;
        out.write(float4(clamp(palette[i].rgb + j, 0.0, 1.0), 1.0), gid);
    }

    // Per-colour binary mask (1 where this region, else 0).
    kernel void region_mask(texture2d<float, access::write> out [[texture(0)]],
                            texture2d<uint, access::read> idx [[texture(1)]],
                            constant uint& target [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        out.write(float4(idx.read(gid).r == target ? 1.0 : 0.0), gid);
    }

    // Pass 2: composite a textured, soft brush of `color` with a luminance gate.
    kernel void brush_composite(texture2d<float, access::write> outCanvas [[texture(0)]],
                                texture2d<float, access::read> inCanvas [[texture(1)]],
                                texture2d<float, access::read> mask [[texture(2)]],
                                texture2d<float, access::read> stencil [[texture(3)]],
                                constant float4* palette [[buffer(0)]],
                                constant uint& colorIndex [[buffer(1)]],
                                constant float& amount [[buffer(2)]],
                                constant float& volumeDiff [[buffer(3)]],
                                constant uint& useStencil [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= outCanvas.get_width() || gid.y >= outCanvas.get_height()) return;
        float3 color = palette[colorIndex].rgb;
        float3 cur = inCanvas.read(gid).rgb;
        float m = mask.read(gid).r;
        if (useStencil) {
            uint2 sc = uint2(gid.x % stencil.get_width(), gid.y % stencil.get_height());
            m *= stencil.read(sc).r;
        }
        float3 painted = cur * (1.0 - m) + color * m;
        float3 cand = cur * (1.0 - amount) + painted * amount;
        float pv = (cur.r + cur.g + cur.b) / 3.0;
        float tv = (cand.r + cand.g + cand.b) / 3.0;
        outCanvas.write(float4(abs(pv - tv) < volumeDiff ? cand : cur, 1.0), gid);
    }

    // Pass 3: reintroduce the original image through textured per-colour masks.
    kernel void comp_masked(texture2d<float, access::write> outCanvas [[texture(0)]],
                            texture2d<float, access::read> inCanvas [[texture(1)]],
                            texture2d<float, access::read> original [[texture(2)]],
                            texture2d<float, access::read> mask [[texture(3)]],
                            texture2d<float, access::read> stencil [[texture(4)]],
                            constant uint& useStencil [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= outCanvas.get_width() || gid.y >= outCanvas.get_height()) return;
        float m = mask.read(gid).r;
        if (useStencil) {
            uint2 sc = uint2(gid.x % stencil.get_width(), gid.y % stencil.get_height());
            m *= stencil.read(sc).r;
        }
        float3 cur = inCanvas.read(gid).rgb;
        float3 orig = original.read(gid).rgb;
        outCanvas.write(float4(m > 0.5 ? orig : cur, 1.0), gid);
    }

    // Clamp a (blurred) mask to a hard edge.
    kernel void mask_clamp(texture2d<float, access::write> out [[texture(0)]],
                           texture2d<float, access::read> src [[texture(1)]],
                           constant float& threshold [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        out.write(float4(src.read(gid).r > threshold ? 1.0 : 0.0), gid);
    }

    // Glow: blend the image toward its blurred self per channel.
    kernel void glow_add(texture2d<float, access::write> out [[texture(0)]],
                         texture2d<float, access::read> image [[texture(1)]],
                         texture2d<float, access::read> blurred [[texture(2)]],
                         constant float& amount [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
        float3 a = image.read(gid).rgb, b = blurred.read(gid).rgb;
        out.write(float4(a * (1.0 - amount) + b * amount, 1.0), gid);
    }
    """
}
