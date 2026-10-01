import AppKit
import SwiftUI
import SwiftMath

/// Bound main-thread parsing and automatic text layout. The full expression
/// remains available through Copy and an explicit, scrollable text disclosure.
enum FormulaSizePolicy {
    static let maximumAutomaticBytes = 8_000
    static let maximumPreviewCharacters = 1_200

    static func isLong(latex: String, expression: String) -> Bool {
        latex.utf8.count > maximumAutomaticBytes || expression.utf8.count > maximumAutomaticBytes
    }

    static func preview(_ expression: String) -> String {
        let prefix = expression.prefix(maximumPreviewCharacters)
        return String(prefix) + (prefix.endIndex == expression.endIndex ? "" : "\n…")
    }
}

struct MathFormula: View {
    let latex: String
    let expression: String
    var fontSize: CGFloat = 22
    var showExpression = true
    @State private var showFullExpression = false

    // Keep the size test before the LaTeX parser: a long expression must never
    // enter SwiftMath merely to determine whether its syntax is supported.
    private var isLong: Bool { FormulaSizePolicy.isLong(latex: latex, expression: expression) }

    var body: some View {
        if isLong {
            VStack(alignment: .leading, spacing: 5) {
                Label("Long expression", systemImage: "text.alignleft").font(.callout.weight(.medium))
                Text("A compact preview is shown. The Copy menu includes the complete expression.")
                    .font(.caption).foregroundStyle(.secondary)
                Text(FormulaSizePolicy.preview(expression))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(8)
                    .padding(.vertical, 6)
                Button(showFullExpression ? "Hide complete expression" : "Show complete expression") {
                    showFullExpression.toggle()
                }
                .buttonStyle(.link)
                .font(.caption)
                if showFullExpression {
                    ReadOnlyExpression(text: expression)
                        .frame(height: 190)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay { RoundedRectangle(cornerRadius: 6).stroke(.separator, lineWidth: 1) }
                }
            }
            .onChange(of: expression) { _, _ in showFullExpression = false }
        } else {
            let supported = MTMathListBuilder.build(fromString: latex) != nil
            VStack(alignment: .leading, spacing: 5) {
                if supported {
                    ScrollView(.horizontal) {
                        NativeMathLabel(latex: latex, fontSize: fontSize)
                            .fixedSize(horizontal: true, vertical: true)
                            .padding(.vertical, 4)
                    }
                    .accessibilityLabel(expression)
                }
                if showExpression || !supported {
                    Text(expression).font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary).textSelection(.enabled)
                }
                if !supported {
                    Label("Shown as text: this formula uses unsupported typesetting.", systemImage: "text.alignleft")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// Compact formula for headings and button labels. A horizontal scroll view
/// would consume clicks and expand a button label; use the natural fitting size.
struct InlineMathFormula: View {
    let latex: String
    var fontSize: CGFloat = 18
    var body: some View {
        NativeMathLabel(latex: latex, fontSize: fontSize)
            .fixedSize(horizontal: true, vertical: true)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}

struct ConventionButton: View {
    var latex: String = ""
    var text: String? = nil
    let accessibilityLabel: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Group {
                if let text { Text(text).font(.system(size: 16, weight: .medium)) }
                else { InlineMathFormula(latex: latex, fontSize: 16) }
            }
                .frame(minHeight: 28)
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .frame(maxWidth: .infinity)
                .background(selected ? Color.accentColor.opacity(0.18) : Color.clear,
                            in: RoundedRectangle(cornerRadius: 7))
                .overlay {
                    RoundedRectangle(cornerRadius: 7)
                        .stroke(selected ? Color.accentColor : Color.secondary.opacity(0.25), lineWidth: selected ? 1.5 : 1)
                }
                .contentShape(RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityValue(selected ? "Selected" : "Not selected")
        .accessibilityAddTraits(selected ? [.isSelected] : [])
    }
}

/// AppKit lays out the visible portion of a long expression as the user scrolls,
/// instead of asking a SwiftUI Text view to size the entire expression at once.
private struct ReadOnlyExpression: NSViewRepresentable {
    let text: String

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        let view = NSTextView()
        view.isEditable = false
        view.isSelectable = true
        view.isRichText = false
        view.usesFindBar = true
        view.isVerticallyResizable = true
        view.isHorizontallyResizable = false
        view.autoresizingMask = [.width]
        view.minSize = .zero
        view.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        view.textContainer?.widthTracksTextView = true
        view.layoutManager?.allowsNonContiguousLayout = true
        view.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        view.textColor = .textColor
        view.backgroundColor = .textBackgroundColor
        view.textContainerInset = NSSize(width: 10, height: 10)
        view.string = text
        view.setAccessibilityLabel("Complete expression")
        scroll.documentView = view
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let view = scroll.documentView as? NSTextView, view.string != text else { return }
        view.string = text
    }
}

private struct NativeMathLabel: NSViewRepresentable {
    let latex: String
    let fontSize: CGFloat
    @Environment(\.colorScheme) private var colorScheme

    func makeNSView(context: Context) -> NativeMathCanvas {
        let label = NativeMathCanvas()
        label.mathLabel.displayErrorInline = false
        label.mathLabel.labelMode = .display
        label.mathLabel.textAlignment = .left
        label.setContentHuggingPriority(.required, for: .horizontal)
        label.setContentHuggingPriority(.required, for: .vertical)
        configure(label)
        return label
    }
    func updateNSView(_ view: NativeMathCanvas, context: Context) { configure(view) }
    func sizeThatFits(_ proposal: ProposedViewSize, nsView: NativeMathCanvas, context: Context) -> CGSize? {
        // SwiftMath exposes fittingSize on macOS. NSView's inherited
        // intrinsicContentSize is (-1, -1), which would clip fractions to 28 pt.
        let size = nsView.fittingSize
        return CGSize(width: max(size.width, 1), height: max(size.height, 28))
    }
    private func configure(_ canvas: NativeMathCanvas) {
        let label = canvas.mathLabel
        label.fontSize = fontSize
        if label.latex != latex { label.latex = latex }
        // SwiftMath resolves its color into CoreGraphics before draw time.
        // Use the SwiftUI appearance explicitly rather than a dynamic NSColor
        // attached to the detached typesetting label's default appearance.
        label.textColor = colorScheme == .dark ? .white : .black
        canvas.invalidateIntrinsicContentSize()
        canvas.needsDisplay = true
    }
}

/// AppKit may supply a flipped text matrix inside a SwiftUI hosting view.
/// SwiftMath positions its CoreText glyphs in an unflipped coordinate system;
/// restore that text matrix explicitly before drawing to keep glyphs upright.
final class NativeMathCanvas: NSView {
    let mathLabel = MTMathUILabel()
    override var isFlipped: Bool { false }
    override var fittingSize: CGSize { mathLabel.fittingSize }
    override var intrinsicContentSize: CGSize { mathLabel.fittingSize }

    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        mathLabel.frame = bounds
        mathLabel.layout()
        context.saveGState()
        context.textMatrix = .identity
        mathLabel.displayList?.draw(context)
        context.restoreGState()
    }
}

func copyToClipboard(_ string: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(string, forType: .string)
}
