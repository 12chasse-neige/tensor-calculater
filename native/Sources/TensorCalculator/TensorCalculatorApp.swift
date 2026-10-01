import AppKit
import SwiftUI

@main
struct TensorCalculatorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    var body: some Scene {
        DocumentGroup(newDocument: CalculationDocument()) { configuration in
            CalculationView(document: configuration.$document)
                .frame(minWidth: 900, minHeight: 680)
        }
        .defaultSize(width: 1180, height: 820)
        .commands {
            CommandGroup(after: .help) {
                Button("Tensor notation and conventions") {
                    NSApplication.shared.sendAction(#selector(AppDelegate.showHelp), to: NSApplication.shared.delegate, from: nil)
                }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }
    @objc func showHelp() {
        let alert = NSAlert()
        alert.messageText = "Tensor calculation"
        alert.informativeText = "Enter a symmetric covariant metric gᵢⱼ in the coordinate order shown. Write one bracketed row per line; use * for multiplication and ^ or ** for powers. Define constants and unknown functions before using them.\n\nComponents use coordinate labels, with index positions shown by superscripts and subscripts. Only components proved to be zero symbolically are omitted. The exact curvature sign convention accompanies each result.\n\nCalculate: ⌘↩   Cancel: ⌘.   Save: ⌘S\n\nChanging the metric keeps the previous result visible until a new calculation completes. Results are marked when their inputs no longer match."
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
