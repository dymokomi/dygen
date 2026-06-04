import Foundation

/// A typed node parameter value. Encodes to a single-key JSON object
/// (e.g. `{"int": 64}`, `{"color": [0.3,0.3,0.3]}`) for a readable scene file.
/// Mirrors dray's `MaterialValue` pattern.
public enum NodeValue: Codable, Equatable {
    case float(Float)
    case int(Int)
    case bool(Bool)
    case color(Float, Float, Float)
    case float3(Float, Float, Float)
    case string(String)

    private enum CodingKeys: String, CodingKey {
        case float, int, bool, color, float3, string
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(Float.self, forKey: .float) { self = .float(v); return }
        if let v = try c.decodeIfPresent(Int.self, forKey: .int) { self = .int(v); return }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .bool) { self = .bool(v); return }
        if let v = try c.decodeIfPresent([Float].self, forKey: .color), v.count == 3 { self = .color(v[0], v[1], v[2]); return }
        if let v = try c.decodeIfPresent([Float].self, forKey: .float3), v.count == 3 { self = .float3(v[0], v[1], v[2]); return }
        if let v = try c.decodeIfPresent(String.self, forKey: .string) { self = .string(v); return }
        throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                                                debugDescription: "Unrecognized NodeValue"))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .float(let v):          try c.encode(v, forKey: .float)
        case .int(let v):            try c.encode(v, forKey: .int)
        case .bool(let v):           try c.encode(v, forKey: .bool)
        case .color(let r, let g, let b):    try c.encode([r, g, b], forKey: .color)
        case .float3(let x, let y, let z):   try c.encode([x, y, z], forKey: .float3)
        case .string(let v):         try c.encode(v, forKey: .string)
        }
    }

    // Convenience accessors (lenient: int<->float coerce).
    public var floatValue: Float? {
        switch self { case .float(let v): return v; case .int(let v): return Float(v); default: return nil }
    }
    public var intValue: Int? {
        switch self { case .int(let v): return v; case .float(let v): return Int(v); default: return nil }
    }
    public var boolValue: Bool? { if case .bool(let v) = self { return v }; return nil }
    public var stringValue: String? { if case .string(let v) = self { return v }; return nil }
    public var rgb: (Float, Float, Float)? {
        switch self { case .color(let r, let g, let b), .float3(let r, let g, let b): return (r, g, b); default: return nil }
    }
}
