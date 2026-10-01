import Foundation
import Combine
import Darwin

/// Each calculation owns its own process. Heavy symbolic operations can always be
/// cancelled without waiting for the mathematics library to yield cooperatively.
@MainActor
final class CalculationModel: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var status = "Ready"
    @Published private(set) var result: CalculationResult?
    @Published private(set) var resultInputs: CalculationInputs?
    @Published private(set) var errorMessage: String?
    @Published private(set) var diagnostic = ""
    private var session: WorkerSession?
    private var activeID: String?
    private var receivedResult = false

    func restore(_ document: CalculationDocument) {
        guard !isRunning else { return }
        result = document.result
        resultInputs = document.resultInputs
        status = result == nil ? "Ready" : "Saved result"
    }

    func calculate(_ inputs: CalculationInputs) {
        guard !isRunning else { return }
        if let message = inputs.validationMessage {
            errorMessage = message
            return
        }
        let id = UUID().uuidString
        activeID = id
        errorMessage = nil
        receivedResult = false
        diagnostic = ""
        status = "Starting calculation…"
        isRunning = true
        do {
            let request = try JSONEncoder().encode(CalculationRequest(id: id, inputs: inputs))
            let worker = try WorkerSession(
                executable: Self.workerURL(),
                request: request,
                event: { [weak self] event in
                    DispatchQueue.main.async {
                        guard let self, self.activeID == id, event.id == id else { return }
                        guard self.errorMessage == nil else { return }
                        guard event.version == 1 else {
                            self.fail("The calculation engine uses an unsupported protocol version.")
                            return
                        }
                        switch event.type {
                        case "progress": self.status = event.message ?? event.stage ?? "Calculating…"
                        case "error": self.fail(event.message ?? "The calculation engine reported an error.")
                        case "result":
                            guard let result = event.result else {
                                self.fail("The calculation engine returned an empty result.")
                                return
                            }
                            self.result = result
                            self.resultInputs = inputs
                            self.receivedResult = true
                            self.status = String(format: "Completed in %.2f s", result.elapsed_seconds)
                        default: break
                        }
                    }
                },
                malformed: { [weak self] message in
                    DispatchQueue.main.async {
                        guard let self, self.activeID == id else { return }
                        self.fail(message)
                    }
                },
                completed: { [weak self] exitCode, diagnostic in
                    DispatchQueue.main.async {
                        guard let self, self.activeID == id else { return }
                        self.diagnostic = diagnostic
                        self.isRunning = false
                        self.session = nil
                        self.activeID = nil
                        if self.errorMessage == nil && (exitCode != 0 || !self.receivedResult) {
                            self.errorMessage = exitCode != 0
                                ? "The calculation engine stopped (exit \(exitCode)). \(diagnostic.trimmingCharacters(in: .whitespacesAndNewlines))"
                                : "The calculation engine finished without a result."
                            self.status = "Calculation failed"
                        }
                    }
                }
            )
            session = worker
            try worker.start()
        } catch {
            session = nil
            activeID = nil
            isRunning = false
            status = "Calculation failed"
            errorMessage = error.localizedDescription
        }
    }

    func cancel() {
        guard isRunning else { return }
        // Invalidate this request before terminating: queued output cannot replace
        // results from the next request, even if the old process exits late.
        activeID = nil
        session?.cancel()
        session = nil
        isRunning = false
        status = "Cancelled"
    }

    func dismissError() { errorMessage = nil }
    private func fail(_ message: String) {
        errorMessage = message
        status = "Calculation failed"
        session?.cancel()
    }

    private static func workerURL() throws -> URL {
        var candidates: [URL] = []
        if let override = ProcessInfo.processInfo.environment["TENSOR_WORKER_PATH"], !override.isEmpty {
            candidates.append(URL(fileURLWithPath: override))
        }
        if let executableDirectory = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(executableDirectory.appendingPathComponent("tensor-worker"))
        }
        if let worker = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0.path) }) { return worker }
        throw NSError(domain: "TensorCalculator", code: 2, userInfo: [NSLocalizedDescriptionKey:
            "The calculation engine is missing from this app. Rebuild the application bundle. For development, set TENSOR_WORKER_PATH to the tensor-worker executable."])
    }
}

/// Reads stdout and stderr on separate queues so neither pipe can fill and
/// deadlock a calculation. The completion callback follows both EOFs.
private final class WorkerSession {
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private let errors = Pipe()
    private let request: Data
    private let event: (WorkerEvent) -> Void
    private let malformed: (String) -> Void
    private let completed: (Int32, String) -> Void
    private let readers = DispatchGroup()
    private let lock = NSLock()
    private var diagnostics = Data()
    private let diagnosticLimit = 64 * 1024
    private let maximumEventBytes = 32 * 1024 * 1024

    init(executable: URL, request: Data, event: @escaping (WorkerEvent) -> Void,
         malformed: @escaping (String) -> Void, completed: @escaping (Int32, String) -> Void) throws {
        self.request = request
        self.event = event
        self.malformed = malformed
        self.completed = completed
        process.executableURL = executable
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors
    }

    func start() throws {
        readers.enter()
        readers.enter()
        process.terminationHandler = { [self] process in
            readers.notify(queue: .global(qos: .userInitiated)) { [self] in
                lock.lock()
                let diagnostic = String(decoding: diagnostics, as: UTF8.self)
                lock.unlock()
                completed(process.terminationStatus, diagnostic)
                process.terminationHandler = nil
            }
        }
        do {
            try process.run()
        } catch {
            readers.leave()
            readers.leave()
            process.terminationHandler = nil
            throw error
        }
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            defer { readers.leave() }
            readEvents()
        }
        DispatchQueue.global(qos: .utility).async { [self] in
            defer { readers.leave() }
            while true {
                let chunk = errors.fileHandleForReading.availableData
                if chunk.isEmpty { break }
                lock.lock()
                diagnostics.append(chunk)
                if diagnostics.count > diagnosticLimit { diagnostics.removeFirst(diagnostics.count - diagnosticLimit) }
                lock.unlock()
            }
        }
        // Avoid SIGPIPE in the GUI if the helper exits before consuming stdin.
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do {
                try input.fileHandleForWriting.write(contentsOf: request + Data([10]))
                try input.fileHandleForWriting.close()
            } catch {
                malformed("Could not send the metric to the calculation engine: \(error.localizedDescription)")
                cancel()
            }
        }
    }

    func cancel() {
        guard process.isRunning else { return }
        process.terminate()
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2) { [self] in
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
    }

    private func readEvents() {
        var buffer = Data()
        let decoder = JSONDecoder()
        while true {
            let chunk = output.fileHandleForReading.availableData
            if chunk.isEmpty { break }
            buffer.append(chunk)
            while let newline = buffer.firstIndex(of: 10) {
                let line = buffer[..<newline]
                buffer.removeSubrange(...newline)
                if line.isEmpty { continue }
                do { event(try decoder.decode(WorkerEvent.self, from: line)) }
                catch {
                    malformed("The calculation engine returned an unreadable response: \(error.localizedDescription)")
                    cancel()
                    return
                }
            }
            if buffer.count > maximumEventBytes {
                malformed("The calculation result exceeded the supported message size.")
                cancel()
                return
            }
        }
        if !buffer.isEmpty {
            do { event(try decoder.decode(WorkerEvent.self, from: buffer)) }
            catch { malformed("The calculation engine returned an incomplete response.") }
        }
    }
}
