use std::collections::HashMap;
use std::process::Command;
use std::env;
use std::fs;
use std::time::Instant;
use rand::Rng;
use itertools::Itertools;
use serde_json::{json, Value};

fn parse_parameters(json: &serde_json::Value) -> Vec<Vec<i64>> {
    let parameters = &json["metadata"]["configuration_space"]["parameters"];
    println!("{}", parameters);

    let parsed_parameters: Vec<Vec<_>> = parameters.as_array().unwrap().iter().map(|item| {
        if let Some(field) = item["values"].get("value_range") {
            let start = field["start"].as_i64().unwrap();
            let end = field["end"].as_i64().unwrap() + (field["inclusive"].as_bool().unwrap() as i64);
            let stride = field["stride"].as_i64().unwrap() as usize;
            let result_range = (start..end).step_by(stride).collect_vec();
            result_range
        } else {
            let result_range = item["values"]["value_list"].as_array().unwrap().iter().map(|it| {
                it.as_i64().unwrap()
            }).collect_vec();
            result_range
        }
    }).collect_vec();
    parsed_parameters
}

fn compile(meta: &serde_json::Value, config_hash: &HashMap<&str, i64>, gpu: i64) -> Value {

    let parallel = false;
    let benchmark_name = &meta["benchmark_suite"]["benchmark_name"].as_str().unwrap();

    let _parallel_cmd = format!(" {} ",  if parallel { "parallel" } else { "default" });
    let mut params_cmd = format!(r#"PARAMETERS=""#);
    // print(cfg)
    for (param_name, param_value) in config_hash {
        params_cmd.push_str(format!(" -D{}={} ", param_name, param_value).as_str());
    }
    params_cmd.push_str(format!(r#"""#).as_str());

    let dir = format!("./{}", benchmark_name);
    let bin_name = format!("{}-{}", benchmark_name, gpu);

    let args_hash: HashMap<&&str, String> = config_hash.iter().map(|(key, value)| {
        (key, value.to_string())
    }).collect();

    let mut args = Vec::new();
    args.push("./script.sh");
    for (_key, value) in args_hash.iter() {
        args.push(value.as_str());
    }
    args.push(bin_name.as_str());

    call_program(Command::new("/bin/sh")
        .args(args)
        .current_dir(dir))
}

fn call_program(cmd: &mut Command) -> Value {
    let now = Instant::now();
    println!("{:?}", cmd);
    let output = cmd.output().expect("failed to execute process");
    let elapsed = now.elapsed();
    let duration = elapsed.as_secs_f64();
    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    let correctness: f64 = { if output.status.success() == true { 1.0 } else { 0.0 } };
    let result = json!({
        "error": String::from_utf8_lossy(&output.stderr),
        "time": duration,
        "correctness": correctness
    });
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    return result;

    //assert!(output.status.success());
}

fn run(_meta: &serde_json::Value) -> Value {
    call_program(Command::new("/bin/sh").args(&["./run-script.sh", "1"]).current_dir("./bfs"))
}

fn pick_config<'a>(i: usize,
                   cartesian_space: &'a Vec<Vec<i64>>,
                   _meta: &serde_json::Value,
                   search_method: fn(&Vec<Vec<i64>>, usize) -> &Vec<i64>)
    -> &'a Vec<i64> {
    search_method(cartesian_space, i)
}

fn brute_force(cartesian_space: &Vec<Vec<i64>>, i: usize) -> &Vec<i64> {
    &cartesian_space[i]
}

fn random_search(cartesian_space: &Vec<Vec<i64>>, _i: usize) -> &Vec<i64> {
    let mut rng = rand::thread_rng();
    let k = rng.gen_range(0..cartesian_space.len());
    &cartesian_space[k]
}

fn get_search_method(meta: &serde_json::Value) -> fn(&Vec<Vec<i64>>, usize) -> &Vec<i64> {
    let technique = meta["search_settings"]["search_technique"]["name"].as_str();

    if technique.unwrap().to_string().as_str() == "brute_force" {
        brute_force
    } else {
        random_search
    }
}

fn build_result(compile_result: &Value, run_result: &Value, config_result: &Value) -> Value {
    let times = json!({
        "kernel": run_result["time"],
        "compile": compile_result["time"]
    });
    let mut invalidity = "none";
    if compile_result["error"].as_str().is_some() && compile_result["error"].to_string().contains("error") {
        invalidity = compile_result["error"].as_str().unwrap();
    } else if run_result["error"].as_str().is_some() {
        invalidity = run_result["error"].as_str().unwrap();
    } else if run_result["error"] == "correctness" {
        invalidity = "correctness";
    }
    let correctness = run_result["correctness"].as_f64().unwrap();
    json!({
        "times": times,
        "configuration": config_result,
        "correctness": correctness,
        "invalidity": invalidity,
    })
}

fn tune(meta: &serde_json::Value, cartesian_space: Vec<Vec<i64>>) -> Vec<Value> {
    //let budget = &meta["search_settings"]["budget"]["steps"].as_i64().unwrap();
    let budget: &i64 = &2;
    let search_method = get_search_method(meta);
    let mut results: Vec<Value> = Vec::new();
    for i in 1..*budget as usize {
        let config = pick_config(i, &cartesian_space, meta, search_method);
        let params = &meta["configuration_space"]["parameters"].as_array().unwrap();
        let mut config_hash = HashMap::new();
        for (i, param) in params.iter().enumerate() {
            config_hash.insert(
                param["name"].as_str().unwrap(),
                config[i]
            );
        }
        let config_json = json!(config_hash);
        let compile_result = compile(meta, &config_hash, 1);
        let run_result = run(meta);
        let result = build_result(&compile_result, &run_result, &config_json);
        println!("{:?}", result);
        results.push(result);
    }
    results
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let data = fs::read_to_string(path).expect("Unable to read file");
    let json: serde_json::Value = serde_json::from_str(data.as_str()).expect("JSON was not well-formatted");
    let parsed_parameters = parse_parameters(&json);
    println!("{:?}", parsed_parameters);

    let cartesian_space: Vec<Vec<i64>> = parsed_parameters.into_iter()
        .map(IntoIterator::into_iter).multi_cartesian_product().collect_vec();
    println!("{:?}", cartesian_space);
    let results = tune(&json["metadata"], cartesian_space);
    fs::write("./rust-results.json", serde_json::to_string_pretty(&results).unwrap())
        .expect("Unable to write file");
}

