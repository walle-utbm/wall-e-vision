#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

namespace py = pybind11;

class SnpeYoloDetector {
public:
    SnpeYoloDetector(const std::string& dlc_path) {
        // Load container
        container_ = zdl::DlContainer::IDlContainer::open(dlc_path);
        if (!container_) {
            const char* errStr = zdl::DlSystem::getLastErrorString();
            throw std::runtime_error(std::string("Error while opening the container file: ") + dlc_path + "\nSNPE Error: " + (errStr ? errStr : "Unknown"));
        }

        // Set up the runtime processor order (DSP first, fallback to CPU)
        zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP;
        if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
            std::cerr << "[WARNING] DSP runtime not available, falling back to CPU." << std::endl;
            runtime = zdl::DlSystem::Runtime_t::CPU;
        }

        zdl::DlSystem::RuntimeList runtimeList;
        runtimeList.add(runtime);
        if (runtime != zdl::DlSystem::Runtime_t::CPU) {
            runtimeList.add(zdl::DlSystem::Runtime_t::CPU);
        }

        // Build the SNPE engine
        zdl::SNPE::SNPEBuilder snpeBuilder(container_.get());
        snpe_ = snpeBuilder.setOutputLayers({})
                   .setRuntimeProcessorOrder(runtimeList)
                   .setUseUserSuppliedBuffers(false)
                   .build();

        if (!snpe_) {
            const char* errStr = zdl::DlSystem::getLastErrorString();
            throw std::runtime_error(std::string("Error while building SNPE object: ") + (errStr ? errStr : "Unknown error"));
        }
    }

    py::dict infer(py::array_t<float> input_image) {
        // Request a buffer info from the Python numpy array
        py::buffer_info buf = input_image.request();
        
        // Typical NCHW or NHWC array from OpenCV/Numpy has 4 dimensions (1, 3, 640, 640)
        if (buf.ndim != 4) {
            throw std::runtime_error("Input must have 4 dimensions (NCHW or NHWC)");
        }

        // Get the input tensor name from SNPE
        const auto& inputNamesOpt = snpe_->getInputTensorNames();
        if (!inputNamesOpt) {
            throw std::runtime_error("Error obtaining input tensor names");
        }
        const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
        if (inputNames.size() == 0) {
            throw std::runtime_error("No input tensors found in the model");
        }
        const char* inputName = inputNames.at(0);

        // Get the required input tensor dimensions
        const auto& inputDimsOpt = snpe_->getInputDimensions(inputName);
        if (!inputDimsOpt) {
            throw std::runtime_error("Error obtaining input dimensions");
        }
        const zdl::DlSystem::TensorShape& inputShape = *inputDimsOpt;

        // Create an ITensor for the input
        std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
        if (!inputTensor) {
            throw std::runtime_error("Failed to create input ITensor");
        }

        // Copy data from numpy array into the ITensor
        float* ptr = static_cast<float*>(buf.ptr);
        std::copy(ptr, ptr + inputTensor->getSize(), inputTensor->begin());

        // Execute the network
        zdl::DlSystem::TensorMap outputTensorMap;
        bool execStatus = snpe_->execute(inputTensor.get(), outputTensorMap);
        if (!execStatus) {
            const char* errStr = zdl::DlSystem::getLastErrorString();
            throw std::runtime_error(std::string("Error while executing the network: ") + (errStr ? errStr : "Unknown error"));
        }

        // Prepare a Python dict for all output tensors
        py::dict results;
        zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
        if (tensorNames.size() == 0) {
            throw std::runtime_error("No output tensors found");
        }
        
        for (size_t t = 0; t < tensorNames.size(); ++t) {
            const char* outputName = tensorNames.at(t);
            zdl::DlSystem::ITensor* outputTensor = outputTensorMap.getTensor(outputName);
            
            if (!outputTensor) {
                continue;
            }

            // Prepare the output numpy array
            zdl::DlSystem::TensorShape outShape = outputTensor->getShape();
            std::vector<py::ssize_t> shape;
            for (size_t d = 0; d < outShape.rank(); ++d) {
                shape.push_back(outShape[d]);
            }
            
            auto result = py::array_t<float>(shape);
            py::buffer_info resultBuf = result.request();
            float* resultPtr = static_cast<float*>(resultBuf.ptr);

            // Copy data from ITensor back to numpy array
            size_t i = 0;
            for (auto it = outputTensor->cbegin(); it != outputTensor->cend(); ++it, ++i) {
                resultPtr[i] = *it;
            }
            
            results[outputName] = result;
        }

        return results;
    }

private:
    std::unique_ptr<zdl::DlContainer::IDlContainer> container_;
    std::unique_ptr<zdl::SNPE::SNPE> snpe_;
};

PYBIND11_MODULE(snpe_native, m) {
    m.doc() = "SNPE PyBind11 wrapper for YOLO inference";

    py::class_<SnpeYoloDetector>(m, "SnpeYoloDetector")
        .def(py::init<const std::string&>(), py::arg("dlc_path"))
        .def("infer", &SnpeYoloDetector::infer, py::arg("input_image"), py::return_value_policy::move, "Run inference on a numpy array");
}
