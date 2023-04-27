#ifdef USE_TOPSINFERENCE
std::string backend_type = "Dorado";
std::string output_names = "";
int device_id = 0;
std::vector<int> cluster_ids = {-1};
int compiled_batchsize = 1;
std::string export_executable = "";
std::string load_executable = "";

std::string str_dynamic_max_shape = "2";
std::string str_dynamic_min_shape = "2";
std::string str_dynamic_input_names = "2";

auto it = provider_options_map.find(type);

if (it != provider_options_map.end()) {
  auto topsinference_provider_options = it->second;
  auto tops_provider_options =
      topsinference_provider_options.find("backend_type");
  if (tops_provider_options != topsinference_provider_options.end()) {
    backend_type = tops_provider_options->second;
  }
  tops_provider_options = topsinference_provider_options.find("output_names");
  if (tops_provider_options != topsinference_provider_options.end()) {
    output_names = tops_provider_options->second;
  }
  tops_provider_options = topsinference_provider_options.find("device");
  if (tops_provider_options != topsinference_provider_options.end()) {
    device_id = std::stoi(tops_provider_options->second);
  }
  tops_provider_options = topsinference_provider_options.find("cluster");
  if (tops_provider_options != topsinference_provider_options.end()) {
    cluster_ids.clear();
    auto str = tops_provider_options->second;
    std::for_each(str.begin(), str.end(), [&cluster_ids](char s) {
      if (s == '-') {
        ORT_THROW("Invalid cluster value, positive integer is allowed");
      } else if (s == ',' or s == ' ') {
      } else {
        cluster_ids.push_back(s - '0');
      }
    });
  }
  tops_provider_options =
      topsinference_provider_options.find("compiled_batchsize");
  if (tops_provider_options != topsinference_provider_options.end()) {
    compiled_batchsize = std::stoi(tops_provider_options->second);
  }
  tops_provider_options =
      topsinference_provider_options.find("export_executable");
  if (tops_provider_options != topsinference_provider_options.end()) {
    export_executable = tops_provider_options->second;
  }
  tops_provider_options =
      topsinference_provider_options.find("load_executable");
  if (tops_provider_options != topsinference_provider_options.end()) {
    load_executable = tops_provider_options->second;
  }

  tops_provider_options =
      topsinference_provider_options.find("dynamic_input_names");
  if (tops_provider_options != topsinference_provider_options.end()) {
    str_dynamic_input_names = tops_provider_options->second;

    tops_provider_options =
        topsinference_provider_options.find("dynamic_min_shape");
  }
  if (tops_provider_options != topsinference_provider_options.end()) {
    str_dynamic_min_shape = tops_provider_options->second;
  }

  tops_provider_options =
      topsinference_provider_options.find("dynamic_max_shape");
  if (tops_provider_options != topsinference_provider_options.end()) {
    str_dynamic_max_shape = tops_provider_options->second;
  }
}

std::cout << "!!!!!!!!!!!dynamic_input_names:" << str_dynamic_input_names
          << std::endl;
std::cout << "!!!!!!!!!!!str_dynamic_min_shape:" << str_dynamic_min_shape
          << std::endl;
std::cout << "!!!!!!!!!!!str_dynamic_max_shape:" << str_dynamic_max_shape
          << std::endl;
auto &logger = logging::LoggingManager::DefaultLogger();
LOGS(logger, WARNING) << "!!!!!!!!!!!dynamic_input_names:" +
                             str_dynamic_input_names;
LOGS(logger, WARNING) << str_dynamic_input_names;
LOGS(logger, WARNING) << "!!!!!!!!!!!str_dynamic_min_shape:" +
                             str_dynamic_min_shape;
LOGS(logger, WARNING) << str_dynamic_min_shape;
LOGS(logger, WARNING) << "!!!!!!!!!!!str_dynamic_max_shape:" +
                             str_dynamic_max_shape;
LOGS(logger, WARNING) << str_dynamic_max_shape;
LOGS(logger, WARNING) << "!!!!!!!!!!!compiled_batchsize:";
LOGS(logger, WARNING) << compiled_batchsize;

std::unordered_map<std::string,
                   std::unordered_map<std::string, std::vector<int>>>
    tops_dynamic_shape = {};

// dynamic_min_shape
std::stringstream s_stream_name(str_dynamic_input_names);
std::stringstream s_stream_min(str_dynamic_min_shape);
std::stringstream s_stream_max(str_dynamic_max_shape);

while (s_stream_name.good()) {
  std::string substr_name;
  std::string substr_min;
  std::string substr_max;

  getline(s_stream_name, substr_name, ';');
  getline(s_stream_min, substr_min, ';');
  getline(s_stream_max, substr_max, ';');

  std::stringstream s_min(substr_min);
  std::stringstream s_max(substr_max);

  std::unordered_map<std::string, std::vector<int>> tops_shape;
  std::vector<int> minvect;
  std::vector<int> maxvect;

  while (s_min.good()) {
    std::string s_a;
    std::string s_b;
    getline(s_min, s_a, ',');
    minvect.push_back(atoi(s_a.c_str()));
    getline(s_max, s_b, ',');
    maxvect.push_back(atoi(s_b.c_str()));
  }
  tops_shape["min_shape"] = minvect;
  tops_shape["max_shape"] = maxvect;
  tops_dynamic_shape[substr_name] = tops_shape;
}

std::for_each(tops_dynamic_shape.begin(), tops_dynamic_shape.end(),
              [](auto &p) {
                std::cout << "p{" << p.first << "}\n";
                std::for_each(p.second.begin(), p.second.end(), [](auto &q) {
                  std::cout << "q{" << q.first << "}\n";

                  std::for_each(q.second.begin(), q.second.end(), [](auto &e) {
                    std::cout << "--" << e << "--"
                              << "}\n";
                  });

                });
              });
exit(0);
return onnxruntime::CreateExecutionProviderFactory_TOPSINFERENCE(
           backend_type.c_str(), output_names.c_str(), device_id, cluster_ids,
           compiled_batchsize, export_executable.c_str(),
           load_executable.c_str())
    ->CreateProvider();
#endif