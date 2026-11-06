#pragma once
#include <filesystem>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace quality {
namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr unsigned char METAL = 0, RESIDUE = 1, UNKNOWN = 2, OUTSIDE = 255;

struct Result {
    Json stats;
    cv::Mat image, mask, evidence, overlay, heat;
};

class Evaluator {
public:
    explicit Evaluator(YAML::Node config);
    Result analyze(const cv::Mat& original, const std::string& name) const;
    Json summarize(const cv::Mat& mask) const;
    const YAML::Node& config() const { return cfg_; }
private:
    YAML::Node cfg_;
    cv::Mat roi(const cv::Size& size, const std::string& name) const;
    Json measure(const cv::Mat& mask, const cv::Size& original) const;
};

YAML::Node load_config(const fs::path& path);
double unix_time();
void atomic_json(const fs::path& path, const Json& value);
Json read_json(const fs::path& path);
void save_result(const fs::path& output, const std::string& name, const Result& result);
void save_report(const fs::path& output, const std::vector<Json>& results);
std::vector<fs::path> image_paths(const fs::path& input);
} // namespace quality
