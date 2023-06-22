// example.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <dxgi.h>

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <exception>
#include <iostream>
#include <numeric>
#include <optional>


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


int main()
{
    const int scale = 4;
    const int channel = 3;
    const int64_t batchSize = 1;

    std::wstring model_path{L"realesr-general-x4v3.onnx"};
    std::string imageFilepath{"sample.jpg"};
    std::string outputFilepath{"sample_upscaled.jpg"};

    ////////////////////////////////////////
    // Enumerate GPUs and pick the first NVIDIA or AMD GPU.
    UINT deviceIndex = 0;
    IDXGIAdapter* pAdapter;
    IDXGIFactory* pFactory;

    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory));        
    while (pFactory->EnumAdapters(deviceIndex, &pAdapter) != DXGI_ERROR_NOT_FOUND)
    {
        DXGI_ADAPTER_DESC desc;
        pAdapter->GetDesc(&desc);

        std::wstring deviceDescription(desc.Description);
        if (deviceDescription.find(L"NVIDIA") != std::string::npos || (deviceDescription.find(L"AMD") != std::string::npos))
        {
            std::wcout << "Found NVIDIA or AMD GPU: " << deviceDescription << ". deviceIndex : " << deviceIndex << std::endl;
            break;
        }
        else
        {
            std::wcout << "Skipping non-NVIDIA or non-AMD GPU: " << deviceDescription << std::endl;
        }

        ++deviceIndex;
    }

    pFactory->Release();

    ////////////////////////////////////////
    // Get API, and setup environment.
    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
    const OrtDmlApi* ortDmlApi;
    ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "DirectML_Direct3D_TensorAllocation_Test"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.

    ////////////////////////////////////////
    // Set model-specific session options.
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // For DML EP
    sessionOptions.DisableMemPattern(); // For DML EP
    ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, deviceIndex);

    Ort::Session session(environment, model_path.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting input batch size to "
            << batchSize << "." << std::endl;
        inputDims.at(0) = batchSize;
    }

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting output batch size to "
            << batchSize << "." << std::endl;
        outputDims.at(0) = batchSize;
    }

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << inputName << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    std::cout << "Output Name: " << outputName << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    cv::Mat imageBGR = cv::imread(imageFilepath, cv::IMREAD_COLOR);
    cv::Mat imageBGRFloat;
    cv::Mat imageRGB;
    cv::Mat blob;


    cv::Mat upscaledImageBGR;
    cv::Mat upscaledImageBGR8U;

    int width = imageBGR.cols;
    int height = imageBGR.rows;

    inputDims.at(2) = height;
    inputDims.at(3) = width;
    outputDims.at(2) = height * scale;
    outputDims.at(3) = width * scale;

    imageBGR.convertTo(imageBGRFloat, CV_32F, 1.0f / 255.0f);

    for (size_t i = 0; i < 10; i++)
    {
        std::vector<cv::Mat> inputImages;
        std::vector<cv::Mat> outputImgs;

        auto start_upscale = std::chrono::high_resolution_clock::now();
        cv::cvtColor(imageBGRFloat, imageRGB, cv::COLOR_BGR2RGB);

        inputImages.push_back(imageRGB);
        blob = cv::dnn::blobFromImages(inputImages);

        size_t inputTensorSize = vectorProduct(inputDims);
        size_t outputTensorSize = vectorProduct(outputDims);

        std::vector<int> outDims{batchSize, channel, height* scale, width* scale};
        cv::Mat upscaledBlob(outDims, CV_32F);

        std::vector<const char*> inputNames{inputName.get()};
        std::vector<const char*> outputNames{outputName.get()};
        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, (float*)blob.data, inputTensorSize, inputDims.data(),
            inputDims.size()));
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, (float*)upscaledBlob.data, outputTensorSize,
            outputDims.data(), outputDims.size()));


        //measure inference time
        auto start = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, 
            inputNames.data(),  inputTensors.data(),  batchSize, 
            outputNames.data(), outputTensors.data(), batchSize);    
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_ms = end - start;

        cv::dnn::imagesFromBlob(upscaledBlob, outputImgs);
        cv::cvtColor(outputImgs[0], upscaledImageBGR, cv::COLOR_RGB2BGR);
        auto end_upscale = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> upscale_ms = end_upscale - start_upscale;

        std::cout << "Inference time: " << inference_ms.count() << " ms" << std::endl;
        std::cout << "Upscale time: " << upscale_ms.count() << " ms" << std::endl;
    }
    upscaledImageBGR.convertTo(upscaledImageBGR8U, CV_8UC3, 255.0f);
    cv::imwrite(outputFilepath, upscaledImageBGR8U);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
