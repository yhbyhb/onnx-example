// example.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <dxgi.h>

#include <onnxruntime_cxx_api.h>

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <unicode/unistr.h>

#include <exception>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), T(1), std::multiplies<T>());
}

// ───────────────────────────────────────────────────────────────
//  UTF-16 (std::wstring)  →  UTF-8 (std::string)
// ───────────────────────────────────────────────────────────────
inline std::string wstring_to_utf8(const std::wstring& wstr)
{
    // Windows: wchar_t is 16-bit, identical to ICU's UChar.
    const auto* src = reinterpret_cast<const UChar*>(wstr.data());
    icu::UnicodeString ustr(false, src, static_cast<int32_t>(wstr.length()));

    std::string utf8;
    ustr.toUTF8String(utf8);            // locale-independent, lossless
    return utf8;
}

// ───────────────────────────────────────────────────────────────
//  UTF-8 (std::string)  →  UTF-16 (std::wstring)
// ───────────────────────────────────────────────────────────────
inline std::wstring utf8_to_wstring(const std::string& utf8)
{
    icu::UnicodeString ustr = icu::UnicodeString::fromUTF8(utf8);
    const auto* buf = ustr.getBuffer(); // UChar*
    return std::wstring(
        reinterpret_cast<const wchar_t*>(buf),
        static_cast<size_t>(ustr.length()));
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

        std::string deviceDesc = wstring_to_utf8(desc.Description);
        if (deviceDesc.find("NVIDIA") != std::string::npos || (deviceDesc.find("AMD") != std::string::npos))
        {
            spdlog::info("Found NVIDIA or AMD GPU: {}. deviceIndex : {}", deviceDesc, deviceIndex);
            break;
        }
        else
        {
            spdlog::warn("Skipping non-NVIDIA or non-AMD GPU: {}. deviceIndex : {}", deviceDesc, deviceIndex);
        }

        ++deviceIndex;
    }

    pFactory->Release();

    auto providers = Ort::GetAvailableProviders();
    for (auto& provider : providers)
    {
        std::cout << "Available provider: " << provider << std::endl;
    }

    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions sessionOptions;

    OrtCUDAProviderOptions o;
    memset(&o, 0, sizeof(o));
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(sessionOptions, &o);
    Ort::Session session(env, model_path.c_str(), sessionOptions);

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
        spdlog::info("Got dynamic batch size. Setting input batch size to {}.", batchSize);
        inputDims.at(0) = batchSize;
    }

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1)
    {
        spdlog::info("Got dynamic batch size. Setting output batch size to {}.", batchSize);
        outputDims.at(0) = batchSize;
    }

    spdlog::info("Number of Input Nodes: {}", numInputNodes);
    spdlog::info("Number of Output Nodes: {}", numOutputNodes);
    spdlog::info("Input Name: {}", inputName.get());
    spdlog::info("Input Type: {}", static_cast<int>(inputType));
    spdlog::info("Input Dimensions: {}", inputDims);
    spdlog::info("Output Name: {}", outputName.get());
    spdlog::info("Output Type: {}", static_cast<int>(outputType));
    spdlog::info("Output Dimensions: {}", outputDims);

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

        spdlog::info("Inference time: {} ms", inference_ms.count());
        spdlog::info("Upscale time: {} ms", upscale_ms.count());
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
