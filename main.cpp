#include <map>
#include <set>
#include <limits>
#include <vector>
#include <fstream>
#include <cassert>
#include <optional>
#include <iostream>
#include <exception>
#include <algorithm>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static std::vector<char> ReadFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static void KeyCallback( GLFWwindow *window, int key, int scancode, int action, int mods )
{
    if ( key == GLFW_KEY_ESCAPE && action == GLFW_PRESS )
    {
        glfwSetWindowShouldClose( window, GLFW_TRUE );
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData )
{
    std::cerr << "VALIDATION LAYER: " << pCallbackData->pMessage << "\n";
    return VK_FALSE;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
VkResult CreateDebugUtilsMessengerEXT( VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                      const VkAllocationCallbacks *pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger )
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if ( func )
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator )
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if ( func )
    {
        func(instance, debugMessenger, pAllocator);
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class HelloTriangleApp
{
public:

    HelloTriangleApp() =  default;
    ~HelloTriangleApp() =  default;

    void Run();

private:

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool IsComplete() const
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    void InitWindow();
    void InitVulkan();
    void CreateInstance();
    void SetupDebugMessenger();
    void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    void CreateSurface();
    void PickPhysicalDevice();
    int  RateDeviceSuitability(VkPhysicalDevice device) const;
    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) const;
    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) const;
    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &avaiableFormats);
    VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> &avaiablePresentModes);
    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void CreateLogicalDevice();
    void CreateSwapChain();
    void CreateImageViews();
    void CreateRenderPass();
    void CreateGraphicsPipelines();
    void CreateFrameBuffers();
    void CreateCommandPool();
    void CreateCommandBuffer();
    void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void CreateSyncObjects();
    VkShaderModule CreateShaderModule(const std::vector<char> &code);
    bool CheckDeviceExtensionSupport(VkPhysicalDevice device) const;
    void MainLoop();
    void DrawFrame();
    void Cleanup();

    bool CheckValidationLayerSupport();
    std::vector<const char*> GetRequiredExtensions();

    GLFWwindow *p_window = nullptr;
    VkInstance  p_vulkanInstance;

    const std::vector<const char*> p_validationLayers = { "VK_LAYER_KHRONOS_validation" };
    const std::vector<const char*> p_deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef NDEBUG
    const bool p_enableValidationLayers = false;
#else
    const bool p_enableValidationLayers = true;
#endif // NDEBUG

    VkDebugUtilsMessengerEXT p_debugMessenger;
    VkPhysicalDevice         p_physicalDevice = VK_NULL_HANDLE;
    VkDevice                 p_device;
    VkQueue                  p_graphicsQueue;
    VkQueue                  p_presentQueue;

    VkSurfaceKHR             p_surface;
    VkSwapchainKHR           p_swapChain;

    std::vector<VkImage>     p_swapChainImages;
    VkFormat                 p_swapChainImageFormat;
    VkExtent2D               p_swapChainExtent;

    std::vector<VkImageView> p_swapChainImageViews;

    VkRenderPass             p_renderPass;
    VkPipelineLayout         p_pipelineLayout;
    VkPipeline               p_graphicsPipeline;

    std::vector<VkFramebuffer> p_swapChainFrameBuffers;
    VkCommandPool            p_commandPool;
    VkCommandBuffer          p_commandBuffer;

    VkSemaphore              p_imageAvailableSemaphore;
    VkSemaphore              p_renderFinishedSemaphore;
    VkFence                  p_inFlightFence;
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::InitWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    int width = 512;
    int height = 512;
    const char *title = "Learn Vulkan";
    p_window = glfwCreateWindow(width, height, title, nullptr, nullptr);

    glfwSetKeyCallback(p_window, KeyCallback);
    if ( !p_window )
    {
        assert(false);
        glfwTerminate();
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::Run()
{
    InitWindow();
    InitVulkan();
    MainLoop();
    Cleanup();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::MainLoop()
{
    while (!glfwWindowShouldClose(p_window))
    {
        glfwPollEvents();
        DrawFrame();
    }

    vkDeviceWaitIdle(p_device);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::DrawFrame()
{
    vkWaitForFences(p_device, 1, &p_inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(p_device, 1, &p_inFlightFence);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(p_device, p_swapChain, UINT64_MAX, p_imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    vkResetCommandBuffer(p_commandBuffer, 0);
    RecordCommandBuffer(p_commandBuffer, imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {p_imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &p_commandBuffer;

    VkSemaphore signalSemaphores[] = {p_renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount  = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(p_graphicsQueue, 1, &submitInfo, p_inFlightFence) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {p_swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.pResults = nullptr;

    vkQueuePresentKHR(p_presentQueue, &presentInfo);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::Cleanup()
{
    vkDestroySemaphore(p_device, p_imageAvailableSemaphore, nullptr);
    vkDestroySemaphore(p_device, p_renderFinishedSemaphore, nullptr);
    vkDestroyFence(p_device, p_inFlightFence, nullptr);

    vkDestroyCommandPool(p_device, p_commandPool, nullptr);

    for ( auto framebuffer : p_swapChainFrameBuffers )
    {
        vkDestroyFramebuffer(p_device, framebuffer, nullptr);
    }

    vkDestroyPipeline(p_device, p_graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(p_device, p_pipelineLayout, nullptr);
    vkDestroyRenderPass(p_device, p_renderPass, nullptr);

    for (auto imageView : p_swapChainImageViews)
    {
        vkDestroyImageView(p_device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(p_device, p_swapChain, nullptr);
    vkDestroyDevice(p_device, nullptr);

    if ( p_enableValidationLayers )
    {
        DestroyDebugUtilsMessengerEXT(p_vulkanInstance, p_debugMessenger, nullptr);
    }

    vkDestroySurfaceKHR(p_vulkanInstance, p_surface, nullptr);
    vkDestroyInstance(p_vulkanInstance, nullptr);

    if ( p_window )
    {
        glfwDestroyWindow(p_window);
    }

    glfwTerminate();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool HelloTriangleApp::CheckValidationLayerSupport()
{
    uint32_t layerCount = 0u;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for ( auto layerName : p_validationLayers )
    {
        bool layerFound = false;

        for ( const auto& layerProperties : availableLayers )
        {
            std::cout << layerProperties.layerName << "\n";
            if ( strcmp(layerName, layerProperties.layerName ) == 0 )
            {
                layerFound = true;
                break;
            }
        }

        if ( !layerFound )
        {
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::InitVulkan()
{
    CreateInstance();
    SetupDebugMessenger();
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
    CreateImageViews();
    CreateRenderPass();
    CreateGraphicsPipelines();
    CreateFrameBuffers();
    CreateCommandPool();
    CreateCommandBuffer();
    CreateSyncObjects();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateInstance()
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

    auto extensions = GetRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if ( p_enableValidationLayers && !CheckValidationLayerSupport())
    {
        throw std::runtime_error("validation layers requested but not available!");
    }

    if ( p_enableValidationLayers )
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(p_validationLayers.size());
        createInfo.ppEnabledLayerNames = p_validationLayers.data();
        PopulateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    }
    else
    {
        createInfo.enabledLayerCount = 0u;
        createInfo.pNext = nullptr;
    }

    if ( vkCreateInstance(&createInfo, nullptr, &p_vulkanInstance) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create instance!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::SetupDebugMessenger()
{
    if ( !p_enableValidationLayers )
        return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    PopulateDebugMessengerCreateInfo(createInfo);

    if ( CreateDebugUtilsMessengerEXT(p_vulkanInstance, &createInfo, nullptr, &p_debugMessenger) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateSurface()
{
    if ( glfwCreateWindowSurface(p_vulkanInstance, p_window, nullptr, &p_surface) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create window surface");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::PickPhysicalDevice()
{
    uint32_t deviceCount = 0u;
    vkEnumeratePhysicalDevices(p_vulkanInstance, &deviceCount, nullptr);
    if ( deviceCount == 0u )
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(p_vulkanInstance, &deviceCount, devices.data());

    std::multimap<int, VkPhysicalDevice> candidates;
    for ( const auto& device : devices )
    {
        int score = RateDeviceSuitability(device);
        candidates.emplace(score, device);
    }

    if ( candidates.rbegin()->first > 0 )
    {
        p_physicalDevice = candidates.rbegin()->second;
    }
    else
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int HelloTriangleApp::RateDeviceSuitability(VkPhysicalDevice device) const
{
    int score = 0;
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    if ( deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU )
    {
        score += 1000;
    }

    score += deviceProperties.limits.maxImageDimension2D;

    if ( !deviceFeatures.geometryShader )
    {
        return 0;
    }

    HelloTriangleApp::QueueFamilyIndices indices = FindQueueFamilies(device);
    bool extensionsSupported = CheckDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if ( extensionsSupported )
    {
        SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    if ( !indices.IsComplete() || !extensionsSupported || !swapChainAdequate )
    {
        return 0;
    }


    return score;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
HelloTriangleApp::QueueFamilyIndices HelloTriangleApp::FindQueueFamilies(VkPhysicalDevice device) const
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0u;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamiliies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamiliies.data());

    int i = 0;
    for ( const auto& queueFamily : queueFamiliies )
    {
        if ( queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT )
        {
            indices.graphicsFamily = i;

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, p_surface, &presentSupport);
            if ( presentSupport )
            {
                indices.presentFamily = i;
            }
        }
    }

    ++i;

    return indices;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
HelloTriangleApp::SwapChainSupportDetails HelloTriangleApp::QuerySwapChainSupport(VkPhysicalDevice device) const
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, p_surface, &details.capabilities);

    uint32_t formatCount = 0u;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, p_surface, &formatCount, nullptr);
    if ( formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, p_surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount = 0u;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, p_surface, &presentModeCount, nullptr);
    if ( presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, p_surface, &presentModeCount, details.presentModes.data());
    }
    return details;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
VkSurfaceFormatKHR HelloTriangleApp::ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if ( availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
             availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
VkPresentModeKHR HelloTriangleApp::ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
{
    for ( const auto& availablePresentMode : availablePresentModes )
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            return availablePresentMode;

    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
VkExtent2D HelloTriangleApp::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(p_window, &width, &height);

    VkExtent2D actualExtent = { static_cast<uint32_t>(width),
                                static_cast<uint32_t>(height) };

    actualExtent.width = std::clamp(actualExtent.width,
                                    capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);

    actualExtent.height = std::clamp(actualExtent.height,
                                    capabilities.minImageExtent.height,
                                    capabilities.maxImageExtent.height);

    return actualExtent;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateLogicalDevice()
{
    QueueFamilyIndices indices = FindQueueFamilies(p_physicalDevice);
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),
                                               indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
        queueCreateInfo.queueCount = 1;

        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

    VkPhysicalDeviceFeatures deviceFeatures{};
    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(p_deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = p_deviceExtensions.data();
    // This is not required with newer implementations of vulkan as we already
    // specified this during vulkan instance creation
    if ( p_enableValidationLayers )
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(p_validationLayers.size());
        createInfo.ppEnabledLayerNames = p_validationLayers.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0u;
    }

    if ( vkCreateDevice(p_physicalDevice, &createInfo, nullptr, &p_device) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create a logical device!");
    }

    vkGetDeviceQueue(p_device, indices.graphicsFamily.value(), 0u, &p_graphicsQueue);
    vkGetDeviceQueue(p_device, indices.presentFamily.value(), 0u, &p_presentQueue);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateSwapChain()
{
    SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(p_physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1u;

    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount )
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = p_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto indices = FindQueueFamilies(p_physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if ( indices.graphicsFamily != indices.presentFamily )
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode  = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if ( vkCreateSwapchainKHR(p_device, &createInfo, nullptr, &p_swapChain) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(p_device, p_swapChain, &imageCount, nullptr);
    p_swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(p_device, p_swapChain, &imageCount, p_swapChainImages.data());

    p_swapChainImageFormat = surfaceFormat.format;
    p_swapChainExtent = extent;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateImageViews()
{
    p_swapChainImageViews.resize(p_swapChainImages.size());
    for ( size_t i = 0; i < p_swapChainImages.size(); ++i )
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = p_swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = p_swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if ( vkCreateImageView(p_device, &createInfo, nullptr, &p_swapChainImageViews[i]) != VK_SUCCESS )
        {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = p_swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if ( vkCreateRenderPass(p_device, &renderPassInfo, nullptr, &p_renderPass) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create render pass!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateGraphicsPipelines()
{
    auto vertShaderCode = ReadFile("shaders/vert.spv");
    auto fragShaderCode = ReadFile("shaders/frag.spv");

    VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    inputAssembly.pNext = nullptr;
    inputAssembly.flags = 0;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)p_swapChainExtent.width;
    viewport.height = (float)p_swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = p_swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1u;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1u;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    if ( vkCreatePipelineLayout(p_device, &pipelineLayoutInfo, nullptr, &p_pipelineLayout) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = p_pipelineLayout;
    pipelineInfo.renderPass = p_renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if ( vkCreateGraphicsPipelines(p_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &p_graphicsPipeline) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(p_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(p_device, vertShaderModule, nullptr);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateFrameBuffers()
{
    p_swapChainFrameBuffers.resize(p_swapChainImageViews.size());

    for ( size_t i = 0; i < p_swapChainImageViews.size(); ++i )
    {
        VkImageView attachments[] = { p_swapChainImageViews[i] };

        VkFramebufferCreateInfo frameBufferInfo{};
        frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        frameBufferInfo.renderPass = p_renderPass;
        frameBufferInfo.attachmentCount = 1;
        frameBufferInfo.pAttachments = attachments;
        frameBufferInfo.width = p_swapChainExtent.width;
        frameBufferInfo.height = p_swapChainExtent.height;
        frameBufferInfo.layers = 1;

        if ( vkCreateFramebuffer(p_device, &frameBufferInfo, nullptr, &p_swapChainFrameBuffers[i]) != VK_SUCCESS )
        {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(p_physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if ( vkCreateCommandPool(p_device, &poolInfo, nullptr, &p_commandPool) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create command pool!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = p_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if ( vkAllocateCommandBuffers(p_device, &allocInfo, &p_commandBuffer) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;

    if ( vkBeginCommandBuffer(p_commandBuffer, &beginInfo) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = p_renderPass;
    renderPassInfo.framebuffer = p_swapChainFrameBuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = p_swapChainExtent;

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(p_commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(p_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p_graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(p_swapChainExtent.width);
    viewport.height = static_cast<float>(p_swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(p_commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = p_swapChainExtent;
    vkCmdSetScissor(p_commandBuffer, 0, 1, &scissor);
    vkCmdDraw(p_commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(p_commandBuffer);

    if ( vkEndCommandBuffer(p_commandBuffer) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to record command buffer!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::CreateSyncObjects()
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if ( vkCreateSemaphore(p_device, &semaphoreInfo, nullptr, &p_imageAvailableSemaphore) != VK_SUCCESS ||
         vkCreateSemaphore(p_device, &semaphoreInfo, nullptr, &p_renderFinishedSemaphore) != VK_SUCCESS ||
         vkCreateFence(p_device, &fenceInfo, nullptr, &p_inFlightFence) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create semaphores!");
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
VkShaderModule HelloTriangleApp::CreateShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if ( vkCreateShaderModule(p_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS )
    {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool HelloTriangleApp::CheckDeviceExtensionSupport(VkPhysicalDevice device) const
{
    uint32_t extensionsCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionsCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionsCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionsCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(p_deviceExtensions.begin(), p_deviceExtensions.end());

    for (const auto& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void HelloTriangleApp::PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT    |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
std::vector<const char*> HelloTriangleApp::GetRequiredExtensions()
{
    uint32_t glfwExtensionsCount = 0u;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionsCount);

    if ( p_enableValidationLayers )
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main(int argc, const char *argv[])
{
    HelloTriangleApp app;

    try
    {
        app.Run();
    }
    catch( const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
