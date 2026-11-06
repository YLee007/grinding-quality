#include "quality.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <csignal>
#include <fstream>
#include <iostream>
#include <set>

namespace fs = quality::fs;
using quality::Json;
namespace {
volatile std::sig_atomic_t stop_requested=0;
void stop(int) {stop_requested=1;}
using Clock=std::chrono::steady_clock;

int worker(const fs::path& settings_path) {
    auto settings=quality::read_json(settings_path);
    auto cfg=quality::load_config(settings["config"].get<std::string>());
    cfg["method"]=settings.at("method").get<std::string>();
    cfg["max_side"]=settings.at("max_side").get<int>();
    cfg["roi"]=YAML::Node(YAML::NodeType::Map);
    for (double x:settings.at("roi").get<std::vector<double>>()) cfg["roi"]["default"].push_back(x);
    quality::Evaluator evaluator(cfg);
    const fs::path ipc=settings.at("ipc").get<std::string>();
    quality::atomic_json(ipc/"quality.ready.json",{{"ready",true},{"backend","cpp_opencv"},{"timestamp_unix_s",quality::unix_time()}});
    std::string line;
    while (!stop_requested && std::getline(std::cin,line)) {
        if (line.empty()) continue;
        const auto started=Clock::now();
        auto job=Json::parse(line);
        const auto id=job.at("id").get<std::string>();
        if (id.size()!=32 || id.find_first_not_of("0123456789abcdef")!=std::string::npos)
            throw std::runtime_error("Invalid job ID");
        Json done={{"id",id},{"backend","cpp_opencv"}};
        try {
            const fs::path out=job.at("output").get<std::string>();
            auto result=evaluator.analyze(cv::imread(job.at("image").get<std::string>()),"camera.jpg");
            result.stats["capture"]=job.at("capture");
            result.stats["schedule_clock_source"]=job.at("clock_source");
            quality::save_result(out,"camera.jpg",result);
            done["ok"]=true;
            for (const auto* key:{"status","residual_fraction_roi_lower_bound","unknown_fraction","evaluable_fraction","analysis_ms"})
                done[key]=result.stats[key];
            done["result"]=(out/"camera.json").string();
        } catch (const std::exception& e) {done["ok"]=false;done["error"]=e.what();}
        done["timestamp_unix_s"]=quality::unix_time();
        done["total_ms"]=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        quality::atomic_json(ipc/(id+".done.json"),done);
        std::cout<<done.dump()<<std::endl;
    }
    return 0;
}

int analyze(const quality::Evaluator& evaluator, const fs::path& input, const fs::path& output) {
    const auto paths=quality::image_paths(input);
    if (paths.empty()) throw std::runtime_error("No input images found: "+input.string());
    std::set<std::string> stems;
    for (const auto& path:paths) {
        auto stem=path.stem().string();
        std::transform(stem.begin(),stem.end(),stem.begin(),[](unsigned char c){return std::tolower(c);});
        if (!stems.insert(stem).second) throw std::runtime_error("Input images must have unique stems");
    }
    fs::create_directories(output);
    std::vector<Json> results;
    for (const auto& path:paths) {
        auto result=evaluator.analyze(cv::imread(path.string()),path.filename().string());
        quality::save_result(output,path.filename().string(),result);
        results.push_back(result.stats);
        std::cout<<path.filename()<<": "<<result.stats["status"]<<" analysis_ms="<<result.stats["analysis_ms"]<<std::endl;
    }
    quality::save_report(output,results);
    std::ofstream config(output/"run_config.yaml");
    config<<evaluator.config();
    std::cout<<"Report: "<<output/"report.html"<<std::endl;
    return 0;
}

int camera(const quality::Evaluator& evaluator, const fs::path& output) {
    const auto cfg=evaluator.config(), c=cfg["camera"];
    const int every=c["analyze_every_n_frames"].as<int>(), save_every=c["save_every_n_results"].as<int>();
    if (every<1 || save_every<1) throw std::runtime_error("Camera intervals must be >=1");
    cv::VideoCapture cap(c["id"].as<int>());
    if (!cap.isOpened()) throw std::runtime_error("Camera could not be opened");
    cap.set(cv::CAP_PROP_FRAME_WIDTH,c["width"].as<int>());
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,c["height"].as<int>());
    const auto out=output/("camera_"+std::to_string(static_cast<long long>(quality::unix_time()*1000000)));
    fs::create_directories(out);
    std::vector<Json> saved;
    int frame_id=0, count=0;
    cv::Mat image;
    try {
        while (!stop_requested) {
            if (!cap.read(image)) throw std::runtime_error("Camera frame acquisition failed");
            // Host receipt time, before analysis; not a hardware exposure timestamp.
            const auto capture_host_time_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (++frame_id%every) continue;
            auto result=evaluator.analyze(image,"camera");
            ++count;
            result.stats["capture_host_time_ns"]=capture_host_time_ns;
            std::cout<<result.stats["status"]<<" residual/ROI="<<result.stats["residual_fraction_roi_lower_bound"]<<std::endl;
            if ((count-1)%save_every==0) {
                const auto name="frame_"+std::to_string(frame_id)+".jpg";
                result.stats["image"]=name;
                quality::save_result(out,name,result);
                saved.push_back(result.stats);
            }
            if (cfg["show_window"].as<bool>()) {
                cv::imshow("Grinding quality (provisional)",result.overlay);
                int key=cv::waitKey(1)&255;
                if (key==27 || key=='q') break;
            }
            if (c["max_results"].as<int>()>0 && count>=c["max_results"].as<int>()) break;
        }
    } catch (...) {quality::save_report(out,saved);throw;}
    quality::save_report(out,saved);
    return 0;
}
} // namespace

int main(int argc,char** argv) {
    try {
        cv::setNumThreads(1);
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
        if (argc<2 || std::string(argv[1])=="--help") {
            std::cout<<"grinding_quality analyze|camera --config config/quality.yaml [--input path] [--output path] [--max-side N] [--grid-px N]\n"
                       "grinding_quality worker --worker-config session/worker_config.json\n";
            return argc<2?1:0;
        }
        const std::string task=argv[1];
        fs::path config="config/quality.yaml",input,output,worker_settings;
        int max_side=0, grid=0;
        for (int i=2;i<argc;++i) {
            const std::string key=argv[i];
            if (i+1>=argc) throw std::runtime_error("Missing argument for "+key);
            const std::string value=argv[++i];
            if (key=="--config") config=value;
            else if (key=="--input") input=value;
            else if (key=="--output") output=value;
            else if (key=="--worker-config") worker_settings=value;
            else if (key=="--max-side") {max_side=std::stoi(value);if(max_side<64)throw std::runtime_error("max-side >=64 required");}
            else if (key=="--grid-px") {grid=std::stoi(value);if(grid<1)throw std::runtime_error("grid-px >=1 required");}
            else throw std::runtime_error("Unknown argument: "+key);
        }
        if (task=="worker") {
            if (worker_settings.empty()) throw std::runtime_error("worker requires --worker-config");
            return worker(worker_settings);
        }
        config=fs::absolute(config);
        auto cfg=quality::load_config(config);
        // Standard repository layout: config/quality.yaml -> repository root.
        const auto base=config.parent_path().parent_path();
        if (input.empty()) {input=cfg["input"].as<std::string>();if(input.is_relative())input=base/input;}
        else input=fs::absolute(input);
        if (output.empty()) {output=cfg["output"].as<std::string>();if(output.is_relative())output=base/output;}
        else output=fs::absolute(output);
        if (max_side) cfg["max_side"]=max_side;
        if (grid) cfg["assessment"]["grid_px"]=grid;
        quality::Evaluator evaluator(cfg);
        if (task=="analyze") return analyze(evaluator,input,output);
        if (task=="camera") return camera(evaluator,output);
        throw std::runtime_error("Unknown task: "+task);
    } catch (const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<std::endl;
        return 1;
    }
}
