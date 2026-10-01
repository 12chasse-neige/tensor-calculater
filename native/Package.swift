// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TensorCalculator",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "TensorCalculator", targets: ["TensorCalculator"])],
    dependencies: [.package(url: "https://github.com/mgriebling/SwiftMath.git", exact: "1.7.3")],
    targets: [
        .executableTarget(
            name: "TensorCalculator",
            dependencies: [.product(name: "SwiftMath", package: "SwiftMath")]
        ),
        .testTarget(name: "TensorCalculatorTests", dependencies: ["TensorCalculator", .product(name: "SwiftMath", package: "SwiftMath")])
    ],
    swiftLanguageVersions: [.v5]
)
