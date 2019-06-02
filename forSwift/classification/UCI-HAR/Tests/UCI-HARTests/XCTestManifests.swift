import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(UCI_HARTests.allTests),
    ]
}
#endif
