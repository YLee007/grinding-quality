#include "quality.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace quality {
namespace {
using Clock = std::chrono::steady_clock;
const std::array<cv::Vec3b, 3> COLORS = {cv::Vec3b(80,185,60), cv::Vec3b(55,65,235), cv::Vec3b(30,200,245)};

float ramp(float x, float lo, float hi) { return std::clamp((x-lo)/(hi-lo), 0.0F, 1.0F); }
// Match Python's round-to-even for resized dimensions, ROIs and reported boxes.
int round_even(double x) { return static_cast<int>(std::nearbyint(x)); }

cv::Rect rect(const YAML::Node& node, const cv::Size& size) {
    if (!node.IsSequence() || node.size() != 4) throw std::runtime_error("ROI needs four coordinates");
    std::array<double,4> v{};
    for (int i=0; i<4; ++i) {
        v[i] = node[i].as<double>();
        if (!std::isfinite(v[i]) || v[i]<0 || v[i]>1) throw std::runtime_error("ROI must be in [0,1]");
    }
    if (v[0]>=v[2] || v[1]>=v[3]) throw std::runtime_error("ROI has no area");
    const int x0=round_even(v[0]*size.width), y0=round_even(v[1]*size.height);
    const int x1=round_even(v[2]*size.width), y1=round_even(v[3]*size.height);
    return cv::Rect(x0,y0,x1-x0,y1-y0);
}

void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) throw std::runtime_error("Cannot save image: "+path.string());
}

std::string escape_html(const std::string& text) {
    std::string out;
    for (char c:text) {
        if (c=='&') out+="&amp;";
        else if (c=='<') out+="&lt;";
        else if (c=='>') out+="&gt;";
        else if (c=='\"') out+="&quot;";
        else out+=c;
    }
    return out;
}
} // namespace

YAML::Node load_config(const fs::path& path) {
    return YAML::LoadFile(path.string());
}

Evaluator::Evaluator(YAML::Node config): cfg_(YAML::Clone(config)) {
    if (cfg_["method"].as<std::string>() != "baseline") throw std::runtime_error("C++ supports baseline only");
    if (cfg_["max_side"].as<int>()<64 || cfg_["assessment"]["grid_px"].as<int>()<1)
        throw std::runtime_error("max_side >=64 and grid_px >=1 required");
    for (const auto* key:{"max_residual_fraction","min_evaluable_fraction"}) {
        double v=cfg_["assessment"][key].as<double>();
        if (!std::isfinite(v) || v<0 || v>1) throw std::runtime_error(std::string(key)+" must be in [0,1]");
    }
    for (const auto* key:{"metal_l_min","metal_chroma_max","dark_l_max","rust_h_min","rust_h_max",
                         "rust_s_min","texture_weight","min_evidence","min_margin"}) {
        if (!std::isfinite(cfg_["baseline"][key].as<double>())) throw std::runtime_error("Nonfinite baseline parameter");
    }
    const auto mm=cfg_["assessment"]["mm_per_pixel_original"];
    if (mm && !mm.IsNull() && (!std::isfinite(mm.as<double>()) || mm.as<double>()<=0))
        throw std::runtime_error("mm_per_pixel_original must be positive or null");
    if (cfg_["assessment"]["min_component_area_px"].as<int>()<1)
        throw std::runtime_error("min_component_area_px must be positive");
    rect(cfg_["roi"]["default"], cv::Size(100,100));
}

cv::Mat Evaluator::roi(const cv::Size& size, const std::string& name) const {
    cv::Mat mask(size, CV_8U, cv::Scalar(0));
    const auto r=cfg_["roi"];
    auto region=YAML::Clone(r["default"]);
    if (r["overrides"] && r["overrides"][name]) region=YAML::Clone(r["overrides"][name]);
    mask(rect(region,size)).setTo(255);
    if (r["exclude_rects"] && r["exclude_rects"][name])
        for (auto excluded:r["exclude_rects"][name]) mask(rect(excluded,size)).setTo(0);
    return mask;
}

Json Evaluator::summarize(const cv::Mat& mask) const {
    int total=0, valid=0, residual=0, unknown=0;
    for (int y=0;y<mask.rows;++y) {
        const auto* row=mask.ptr<unsigned char>(y);
        for (int x=0;x<mask.cols;++x) {
            const auto v=row[x];
            total+=v!=OUTSIDE;
            valid+=(v==METAL || v==RESIDUE);
            residual+=v==RESIDUE;
            unknown+=v==UNKNOWN;
        }
    }
    const double coverage=total ? static_cast<double>(valid)/total:0;
    const double lower=total ? static_cast<double>(residual)/total:0;
    const double ratio=valid ? static_cast<double>(residual)/valid:0;
    const auto a=cfg_["assessment"];
    std::string status;
    if (total && lower>a["max_residual_fraction"].as<double>()) status="rework_candidate";
    else if (!valid || coverage<a["min_evaluable_fraction"].as<double>()) status="review";
    else if (ratio>a["max_residual_fraction"].as<double>()) status="rework_candidate";
    else status="clean_candidate";
    return {{"roi_pixels",total},{"evaluable_pixels",valid},{"residual_pixels",residual},
            {"unknown_pixels",unknown},{"evaluable_fraction",coverage},
            {"residual_fraction_evaluable",valid?Json(ratio):Json(nullptr)},
            {"residual_fraction_roi_lower_bound",lower},
            {"unknown_fraction",total?Json(static_cast<double>(unknown)/total):Json(nullptr)},
            {"status",status}};
}

Json Evaluator::measure(const cv::Mat& mask, const cv::Size& original) const {
    Json result=summarize(mask), cells=Json::array(), defects=Json::array();
    const auto a=cfg_["assessment"];
    const int grid=a["grid_px"].as<int>();
    const double sx=static_cast<double>(original.width)/mask.cols, sy=static_cast<double>(original.height)/mask.rows;
    for (int y=0;y<mask.rows;y+=grid) for (int x=0;x<mask.cols;x+=grid) {
        const int right=std::min(x+grid,mask.cols), bottom=std::min(y+grid,mask.rows);
        auto cell=summarize(mask(cv::Rect(x,y,right-x,bottom-y)));
        if (cell["roi_pixels"].get<int>()) {
            cell["box_xyxy_working"]={x,y,right,bottom};
            cells.push_back(cell);
        }
    }
    cv::Mat labels, stats, centers;
    int n=cv::connectedComponentsWithStats(mask==RESIDUE,labels,stats,centers,8,CV_32S);
    for (int i=1;i<n;++i) {
        const int area=stats.at<int>(i,cv::CC_STAT_AREA);
        if (area<a["min_component_area_px"].as<int>()) continue;
        const int x=stats.at<int>(i,cv::CC_STAT_LEFT), y=stats.at<int>(i,cv::CC_STAT_TOP);
        const int w=stats.at<int>(i,cv::CC_STAT_WIDTH), h=stats.at<int>(i,cv::CC_STAT_HEIGHT);
        Json item={{"area_px_original",area*sx*sy},
                   {"box_xyxy_original",{round_even(x*sx),round_even(y*sy),round_even((x+w)*sx),round_even((y+h)*sy)}},
                   {"centroid_xy_original",{centers.at<double>(i,0)*sx,centers.at<double>(i,1)*sy}}};
        const auto mm=a["mm_per_pixel_original"];
        if (mm && !mm.IsNull()) item["area_mm2"]=area*sx*sy*std::pow(mm.as<double>(),2);
        defects.push_back(item);
    }
    std::stable_sort(defects.begin(),defects.end(),[](const Json& a,const Json& b) {
        return a["area_px_original"].get<double>()>b["area_px_original"].get<double>();
    });
    result["cells"]=cells;
    result["residual_components"]=defects;
    result["largest_residual_area_px_original"]=defects.empty()?Json(0):defects[0]["area_px_original"];
    result["coordinate_frame"]="original image pixels; NOT robot/world coordinates";
    result["original_size_wh"]={original.width,original.height};
    result["working_size_wh"]={mask.cols,mask.rows};
    return result;
}

Result Evaluator::analyze(const cv::Mat& original, const std::string& name) const {
    if (original.empty() || original.type()!=CV_8UC3) throw std::runtime_error("Expected nonempty BGR image");
    const auto started=Clock::now();
    Result result;
    double scale=std::min(1.0,static_cast<double>(cfg_["max_side"].as<int>())/std::max(original.rows,original.cols));
    if (scale==1.0) result.image=original.clone();
    else cv::resize(original,result.image,cv::Size(round_even(original.cols*scale),round_even(original.rows*scale)),0,0,cv::INTER_AREA);
    const auto& image=result.image;
    auto roi_mask=roi(image.size(),name);
    cv::Mat lab, hsv, gray8, gray, dx, dy, xx, yy, xy;
    cv::cvtColor(image,lab,cv::COLOR_BGR2Lab);
    cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
    cv::cvtColor(image,gray8,cv::COLOR_BGR2GRAY);
    gray8.convertTo(gray,CV_32F);
    gray/=255.0F;
    cv::Sobel(gray,dx,CV_32F,1,0,3);
    cv::Sobel(gray,dy,CV_32F,0,1,3);
    cv::GaussianBlur(dx.mul(dx),xx,cv::Size(),3);
    cv::GaussianBlur(dy.mul(dy),yy,cv::Size(),3);
    cv::GaussianBlur(dx.mul(dy),xy,cv::Size(),3);
    // Only compute features used by baseline, not unused RF feature stacks.
    const auto b=cfg_["baseline"], g=cfg_["image_gate"];
    const float lmin=b["metal_l_min"].as<float>(), cmax=b["metal_chroma_max"].as<float>();
    const float darkmax=b["dark_l_max"].as<float>(), smin=b["rust_s_min"].as<float>();
    const float hmin=b["rust_h_min"].as<float>(), hmax=b["rust_h_max"].as<float>();
    const float wt=b["texture_weight"].as<float>(), min_evidence=b["min_evidence"].as<float>();
    const float margin=b["min_margin"].as<float>();
    const int over=g["overexposed_min"].as<int>(), under=g["underexposed_max"].as<int>();
    result.mask.create(image.size(),CV_8U);
    result.evidence.create(image.size(),CV_32F);
    for (int y=0;y<image.rows;++y) {
        const auto *lr=lab.ptr<cv::Vec3b>(y), *hr=hsv.ptr<cv::Vec3b>(y), *ir=image.ptr<cv::Vec3b>(y);
        const auto *xr=xx.ptr<float>(y), *yr=yy.ptr<float>(y), *zr=xy.ptr<float>(y);
        const auto *rr=roi_mask.ptr<unsigned char>(y);
        auto *mr=result.mask.ptr<unsigned char>(y);
        auto *er=result.evidence.ptr<float>(y);
        for (int x=0;x<image.cols;++x) {
            const float d=xr[x]-yr[x];
            const float coherence=std::sqrt(d*d+4.0F*zr[x]*zr[x])/(xr[x]+yr[x]+1e-6F);
            const float ca=lr[x][1]-128.0F, cb=lr[x][2]-128.0F;
            const float chroma=std::sqrt(ca*ca+cb*cb), light=lr[x][0];
            const float bright=ramp(light,lmin-35,lmin+35);
            const float neutral=1.0F-ramp(chroma,cmax-7,cmax+10);
            const float metal=bright*neutral*((1.0F-wt)+wt*coherence);
            const bool warm=hr[x][0]>=hmin && hr[x][0]<=hmax;
            const float rust=(warm?1.0F:0.0F)*ramp(hr[x][1],smin-15,smin+35);
            const float dark=1.0F-ramp(light,darkmax-30,darkmax+30);
            const float residue=std::max(rust,0.85F*dark);
            er[x]=std::max(metal,residue);
            mr[x]=metal>=residue?METAL:RESIDUE;
            if (er[x]<min_evidence || std::abs(metal-residue)<margin) mr[x]=UNKNOWN;
            const auto rgb=ir[x];
            if (std::min({rgb[0],rgb[1],rgb[2]})>=over || std::max({rgb[0],rgb[1],rgb[2]})<=under) mr[x]=UNKNOWN;
            if (!rr[x]) mr[x]=OUTSIDE;
        }
    }
    result.stats=measure(result.mask,original.size());
    result.stats.update({{"image",name},{"method","baseline"},{"backend","cpp_opencv"},
        {"analysis_ms",std::chrono::duration<double,std::milli>(Clock::now()-started).count()},
        {"decision_is_provisional",true},{"acceptance_thresholds_are_examples",true},
        {"roi_purpose","exploratory plate inspection; confirm production footprint before use"},
        {"confidence_kind","heuristic evidence, NOT calibrated probability"}});
    result.overlay=image.clone(); result.heat=image.clone();
    for (int y=0;y<image.rows;++y) {
        const auto* mr=result.mask.ptr<unsigned char>(y);
        auto* overlay=result.overlay.ptr<cv::Vec3b>(y);
        const auto* ir=image.ptr<cv::Vec3b>(y);
        for (int x=0;x<image.cols;++x) for (int c=0;c<3;++c)
            overlay[x][c]=static_cast<unsigned char>(mr[x]==OUTSIDE?ir[x][c]*0.35:ir[x][c]*0.55+COLORS[mr[x]][c]*0.45);
    }
    for (const auto& cell:result.stats["cells"]) {
        auto box=cell["box_xyxy_working"].get<std::array<int,4>>();
        const auto status=cell["status"].get<std::string>();
        const auto color=COLORS[status=="clean_candidate"?METAL:status=="rework_candidate"?RESIDUE:UNKNOWN];
        for (int y=box[1];y<box[3];++y) {
            auto* row=result.heat.ptr<cv::Vec3b>(y);
            auto* mask=result.mask.ptr<unsigned char>(y);
            for (int x=box[0];x<box[2];++x) if (mask[x]!=OUTSIDE)
                for (int c=0;c<3;++c) row[x][c]=static_cast<unsigned char>(row[x][c]*0.5+color[c]*0.5);
        }
        cv::rectangle(result.heat,cv::Point(box[0],box[1]),cv::Point(box[2]-1,box[3]-1),cv::Scalar(80,80,80),1);
    }
    return result;
}

double unix_time() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

Json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open JSON: "+path.string());
    Json data; stream>>data; return data;
}

void atomic_json(const fs::path& path, const Json& value) {
    auto temp=path; temp+=".tmp";
    { std::ofstream stream(temp); stream<<value.dump(2); stream.close();
      if (!stream) throw std::runtime_error("Cannot write JSON: "+temp.string()); }
    fs::rename(temp,path);
}

void save_result(const fs::path& out, const std::string& name, const Result& r) {
    fs::create_directories(out);
    const auto stem=fs::path(name).stem().string();
    write_image(out/(stem+"_original.jpg"),r.image);
    write_image(out/(stem+"_overlay.jpg"),r.overlay);
    write_image(out/(stem+"_heatmap.jpg"),r.heat);
    write_image(out/(stem+"_mask.png"),r.mask);
    cv::Mat evidence(r.evidence.size(),CV_8U);
    for (int y=0;y<evidence.rows;++y) for (int x=0;x<evidence.cols;++x)
        evidence.at<unsigned char>(y,x)=static_cast<unsigned char>(std::clamp(r.evidence.at<float>(y,x),0.0F,1.0F)*255.0F);
    write_image(out/(stem+"_evidence.png"),evidence);
    atomic_json(out/(stem+".json"),r.stats);
}

std::vector<fs::path> image_paths(const fs::path& input) {
    std::vector<fs::path> result;
    auto add=[&](const fs::path& p) {
        if (!fs::is_regular_file(p)) return;
        auto ext=p.extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return std::tolower(c);});
        for (const auto* e:{".jpg",".jpeg",".png",".bmp",".tif",".tiff"}) if (ext==e) {result.push_back(p);break;}
    };
    if (fs::is_regular_file(input)) add(input);
    else if (fs::is_directory(input)) for (const auto& p:fs::directory_iterator(input)) add(p.path());
    std::sort(result.begin(),result.end());
    return result;
}

void save_report(const fs::path& out, const std::vector<Json>& results) {
    std::ofstream csv(out/"summary.csv"), html(out/"report.html");
    csv<<"image,method,status,evaluable_fraction,unknown_fraction,residual_fraction_evaluable,residual_fraction_roi_lower_bound,largest_residual_area_px_original,analysis_ms\n";
    html<<u8"<!doctype html><meta charset='utf-8'><title>打磨质量评估</title><style>body{font:16px sans-serif;background:#eef2f7;color:#243349;margin:30px}article{background:white;padding:20px;margin:20px 0;border-radius:10px}.images{display:flex;gap:12px}img{width:31%;object-fit:contain}a{color:#236bb3}</style><h1>打磨质量评估 · OpenCV C++</h1><p>规则法候选结果，非验收结论。绿：金属候选；红：残留候选；黄：待复核。坐标为原图像素，不是机器人坐标。</p>";
    for (const auto& r:results) {
        const auto name=r["image"].get<std::string>(), stem=escape_html(fs::path(name).stem().string());
        std::string csv_name=name;
        for (std::size_t p=0;(p=csv_name.find('"',p))!=std::string::npos;p+=2) csv_name.insert(p,"\"");
        csv<<'"'<<csv_name<<'"';
        for (const auto* key:{"method","status","evaluable_fraction","unknown_fraction","residual_fraction_evaluable",
                             "residual_fraction_roi_lower_bound","largest_residual_area_px_original","analysis_ms"}) {
            csv<<',';
            if (r[key].is_string()) csv<<r[key].get<std::string>();
            else if (!r[key].is_null()) csv<<r[key].dump();
        }
        csv<<'\n';
        html<<"<article><h2>"<<escape_html(name)<<"</h2><p>"<<r["status"].get<std::string>()
            <<" · residual/ROI: "<<std::fixed<<std::setprecision(2)<<r["residual_fraction_roi_lower_bound"].get<double>()*100<<"% · unknown: ";
        if (r["unknown_fraction"].is_null()) html<<"N/A";
        else html<<r["unknown_fraction"].get<double>()*100<<'%';
        html<<"</p><div class='images'>";
        for (const auto* key:{"original","overlay","heatmap"}) html<<"<img src='"<<stem<<'_'<<key<<".jpg'>";
        html<<"</div><a href='"<<stem<<".json'>JSON</a></article>";
    }
    if (!html || !csv) throw std::runtime_error("Report write failed");
}
} // namespace quality
