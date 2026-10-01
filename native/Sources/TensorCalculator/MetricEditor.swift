import AppKit
import SwiftUI

/// The metric is a genuine AppKit text editor: undo, selection, find, scrolling,
/// and keyboard navigation use the same machinery as other Mac document apps.
struct MetricEditor: NSViewRepresentable {
    @Binding var text: String
    @Environment(\.colorScheme) private var colorScheme

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = true
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        scroll.drawsBackground = true
        scroll.backgroundColor = .textBackgroundColor

        let view = NSTextView()
        view.delegate = context.coordinator
        view.isRichText = false
        view.isEditable = true
        view.isSelectable = true
        view.allowsUndo = true
        view.usesFindBar = true
        view.isAutomaticQuoteSubstitutionEnabled = false
        view.isAutomaticDashSubstitutionEnabled = false
        view.isAutomaticTextReplacementEnabled = false
        view.isAutomaticSpellingCorrectionEnabled = false
        view.isContinuousSpellCheckingEnabled = false
        view.isGrammarCheckingEnabled = false
        view.isAutomaticLinkDetectionEnabled = false
        view.font = .monospacedSystemFont(ofSize: 14, weight: .regular)
        view.textColor = .textColor
        view.backgroundColor = .textBackgroundColor
        view.insertionPointColor = .textColor
        view.textContainerInset = NSSize(width: 14, height: 14)
        view.isVerticallyResizable = true
        view.isHorizontallyResizable = true
        view.minSize = NSSize(width: 0, height: 0)
        view.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        view.autoresizingMask = []
        view.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        view.textContainer?.widthTracksTextView = false
        view.string = text
        view.setAccessibilityLabel("Metric matrix")
        scroll.documentView = view
        context.coordinator.highlight(view)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let view = scroll.documentView as? NSTextView else { return }
        if view.string != text {
            view.string = text
            view.undoManager?.removeAllActions()
        }
        context.coordinator.highlight(view)
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: MetricEditor
        private var highlighting = false
        init(_ parent: MetricEditor) { self.parent = parent }

        func textDidChange(_ notification: Notification) {
            guard !highlighting, let view = notification.object as? NSTextView else { return }
            parent.text = view.string
            highlight(view)
        }

        func highlight(_ view: NSTextView) {
            guard let storage = view.textStorage, !highlighting else { return }
            highlighting = true
            defer { highlighting = false }
            let range = NSRange(location: 0, length: (view.string as NSString).length)
            storage.beginEditing()
            storage.addAttributes([.foregroundColor: NSColor.textColor, .font: NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)], range: range)
            let patterns: [(String, NSColor)] = [
                ("\\b(?:sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|sqrt|abs)\\b", .systemPurple),
                ("\\b[0-9]+(?:\\.[0-9]+)?\\b", .systemBlue),
                ("[\\[\\](),]", .secondaryLabelColor)
            ]
            for (pattern, color) in patterns {
                guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
                for match in regex.matches(in: view.string, range: range) {
                    storage.addAttribute(.foregroundColor, value: color, range: match.range)
                }
            }
            storage.endEditing()
            view.typingAttributes = [.font: NSFont.monospacedSystemFont(ofSize: 14, weight: .regular), .foregroundColor: NSColor.textColor]
        }
    }
}
