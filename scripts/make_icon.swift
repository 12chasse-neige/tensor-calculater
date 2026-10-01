// Build-time AppKit drawing. No generated image or font is fetched from a server.
import AppKit
import Foundation

guard CommandLine.arguments.count == 2 else {
    fatalError("Usage: swift make_icon.swift <AppIcon.iconset>")
}
let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

func renderIcon(size: Int, name: String) throws {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
                                 bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                 isPlanar: false, colorSpaceName: .deviceRGB,
                                 bytesPerRow: 0, bitsPerPixel: 0)!
    let context = NSGraphicsContext(bitmapImageRep: bitmap)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    context.imageInterpolation = .high
    let scale = CGFloat(size) / 1024
    context.cgContext.scaleBy(x: scale, y: scale)

    let rect = NSRect(x: 62, y: 62, width: 900, height: 900)
    let outline = NSBezierPath(roundedRect: rect, xRadius: 192, yRadius: 192)
    let shadow = NSShadow()
    shadow.shadowColor = NSColor.black.withAlphaComponent(0.28)
    shadow.shadowBlurRadius = 24
    shadow.shadowOffset = NSSize(width: 0, height: -12)
    shadow.set()
    NSColor(calibratedRed: 0.10, green: 0.23, blue: 0.42, alpha: 1).setFill()
    outline.fill()
    NSShadow().set()
    NSGradient(starting: NSColor(calibratedRed: 0.16, green: 0.42, blue: 0.68, alpha: 1),
               ending: NSColor(calibratedRed: 0.06, green: 0.14, blue: 0.31, alpha: 1))!
        .draw(in: outline, angle: -60)

    // Coordinate grid suggests a metric matrix while the Greek symbol remains
    // legible at small Dock sizes.
    NSColor.white.withAlphaComponent(0.10).setStroke()
    let grid = NSBezierPath()
    grid.lineWidth = 5
    for coordinate in stride(from: CGFloat(248), through: CGFloat(776), by: CGFloat(176)) {
        grid.move(to: NSPoint(x: coordinate, y: 224))
        grid.line(to: NSPoint(x: coordinate, y: 800))
        grid.move(to: NSPoint(x: 224, y: coordinate))
        grid.line(to: NSPoint(x: 800, y: coordinate))
    }
    grid.stroke()

    let symbol = NSAttributedString(string: "Γ", attributes: [
        .font: NSFont.systemFont(ofSize: 580, weight: .medium),
        .foregroundColor: NSColor.white
    ])
    let symbolSize = symbol.size()
    symbol.draw(at: NSPoint(x: (1024 - symbolSize.width) / 2 - 18,
                           y: (1024 - symbolSize.height) / 2 + 18))
    let dot = NSBezierPath(ovalIn: NSRect(x: 656, y: 266, width: 82, height: 82))
    NSColor(calibratedRed: 0.44, green: 0.83, blue: 0.95, alpha: 1).setFill()
    dot.fill()
    NSGraphicsContext.restoreGraphicsState()

    let png = bitmap.representation(using: .png, properties: [:])!
    try png.write(to: output.appendingPathComponent(name))
}

for points in [16, 32, 128, 256, 512] {
    try renderIcon(size: points, name: "icon_\(points)x\(points).png")
    try renderIcon(size: points * 2, name: "icon_\(points)x\(points)@2x.png")
}
