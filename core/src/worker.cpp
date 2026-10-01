#include "tensor/core.hpp"
#include <csignal>
#include <iostream>
#include <stdexcept>

namespace {
void cancelled(int) { tensor::request_cancel(); }
void emit(tensor::JSON event) { std::cout << event.dump() << '\n' << std::flush; }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << "TensorCore 0.1.0 / SymEngine 0.14.0 / protocol 1\n"; return 0;
    }
    std::signal(SIGTERM,cancelled); std::signal(SIGINT,cancelled);
    std::string line;
    while (std::getline(std::cin,line)) {
        if (line.empty()) continue;
        std::string id;
        try {
            if (line.size() > 1024*1024) throw std::runtime_error("Calculation input exceeds 1 MiB.");
            auto request = tensor::JSON::parse(line);
            if (request.is_object() && request.contains("id") && request["id"].is_string()) id = request["id"].get<std::string>();
            auto input = tensor::parse_input(request);
            auto result = tensor::calculate(input,[&](const std::string& stage,const std::string& message) {
                emit({{"version",1},{"id",id},{"type","progress"},{"stage",stage},{"message",message}});
            });
            emit({{"version",1},{"id",id},{"type","result"},{"result",std::move(result)}});
        } catch (const std::exception& e) {
            emit({{"version",1},{"id",id},{"type","error"},{"message",e.what()}});
        }
        if (tensor::cancellation_requested()) break;
    }
    return 0;
}
