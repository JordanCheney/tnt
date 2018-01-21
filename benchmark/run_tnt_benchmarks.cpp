#include <tnt_benchmark_harness.hpp>

#include <tnt/core/errors.hpp>

#include <numeric>
#include <regex>
#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace tnt_bench;

// ----------------------------------------------------------------------------
// Split a string along a delimiter

inline vector<string> split(const string& str,
                            const string& delimiter)
{
    vector<string> strings;

    string::size_type pos = 0, prev = 0;
    while ((pos = str.find(delimiter, prev)) != string::npos) {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

// ----------------------------------------------------------------------------
// Return a helpful message

string help_message()
{
    return string("") +
           "Execute the TNT benchmarking suite\n\n"                                      +
           "==== Filter ====\n"                                                          +
           "-name arg [arg] [arg] ... | Run all benchmarks matching the given name(s)\n" +
           "-type arg [arg] [arg] ... | Run all benchmarks for the given type(s)\n"      +
           "-regex arg                | Run all benchmarks matching the regular expresion";
}

// ----------------------------------------------------------------------------
// Filter benchmarks based on a set of regular expressions

vector<shared_ptr<Benchmark>> filter_benchmarks(const vector<shared_ptr<Benchmark>>& benchmarks,
                                                const vector<regex>& regexs)
{
    vector<shared_ptr<Benchmark>> filtered_benchmarks;
    for (const shared_ptr<Benchmark>& benchmark : benchmarks) {
        bool matched_all = true;
        for (const regex& rex : regexs) {
            matched_all &= regex_match(benchmark->name(), rex, regex_constants::match_any);
            if (!matched_all)
                break;
        }

        if (matched_all)
            filtered_benchmarks.push_back(benchmark);
    }

    return filtered_benchmarks;
}

// ----------------------------------------------------------------------------
// Run a collection of benchmarks and display output about them

static inline string shape_to_string(const tnt::Shape& shape)
{
    stringstream stream;
    stream << shape;
    return stream.str().substr(7, string::npos);
}

void print_table_break(const vector<size_t>& column_sizes)
{
    for (size_t size : column_sizes)
        cout << "+" << string(size + 2, '-');
    cout << "+" << endl;
}

void print_results(const std::unordered_map<std::string, std::vector<BenchmarkResult>>& results,
                   const vector<size_t>& column_sizes)
{
    // Header
    print_table_break(column_sizes);

    for (auto entry : results) {
        for (const BenchmarkResult& result : entry.second) {
            cout << "| " << setw(3) << left << result.vendor          << " | "
                         << setw(4) << left << result.type            << " | "
                         << setw(4) << left << result.instruction_set << " |";

            for (size_t i = 0; i < result.times.size(); ++i)
                cout << " " << setw(column_sizes[i]) << left << setprecision(3) << result.times[i] * 10e-6 << " |";
            cout << endl;
        }
    }
}

int run_benchmarks(const vector<shared_ptr<Benchmark>>& benchmarks)
{
    tnt::print("\nTNT Unit benchmarks:\n\n");

    for (const std::shared_ptr<Benchmark>& benchmark : benchmarks) {
        std::vector<size_t> column_sizes{3, 4, 4};
        for (const tnt::Shape& shape : benchmark->shapes())
            column_sizes.push_back(max(shape_to_string(shape).size(), 8ul));

        // Print the table name
        size_t table_width = accumulate(column_sizes.begin(), column_sizes.end(), 0);
        cout << string((table_width / 2), ' ') << benchmark->name() << endl;

        // Start of the table
        print_table_break(column_sizes);

        // Print the table header
        cout << "| Ven | Type | Inst |";
        for (size_t i = 0; i < benchmark->shapes().size(); ++i)
            cout << " " << setw(column_sizes[i + 3]) << left << shape_to_string(benchmark->shapes()[i]) << " |";
        cout << endl;

        print_results(benchmark->run(), column_sizes);

        // End of the table
        print_table_break(column_sizes);
    }

    return 0;
}

void check(bool cond, const string& msg)
{
    if (!cond) {
        tnt::print("Parse Error: {}\n\n{}", msg, help_message());
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------------------------
// Main function- parse commmand line options and pass the appropriate benchmark
// subset to run_benchmarks()

int main(int argc, char* argv[])
{
    vector<shared_ptr<Benchmark>> all_benchmarks = benchmark_registry();

    // Remove the program name
    argv = &argv[1];
    --argc;

    if (argc == 0)
        return run_benchmarks(all_benchmarks);

    vector<regex> regexs;
    while (argc > 0) {
        const char* fun = argv[0];
        if (fun[0] == '-')
            ++fun;

        int parc = 0;
        while ((parc + 1 < argc) && (argv[parc + 1][0] != '-'))
            ++parc;

        const char** parv = (const char**) &argv[1];
        argc -= parc + 1;
        argv = &argv[parc + 1];

        if (!strcmp(fun, "name")) {
            check(parc > 0, "-name requires at least one argument.");
            for (int i = 0; i < parc; ++i)
                regexs.push_back(regex(parv[i]));
        } else if (!strcmp(fun, "type")) {
            check(parc > 0, "-type requires at least one argument.");
            for (int i = 0; i < parc; ++i)
                regexs.push_back(regex(parv[i]));
        } else if (!strcmp(fun, "regex")) {
            check(parc == 1, "-regex requires a single argument.");
            regexs.push_back(regex(parv[0]));
        } else {
            check(false, "Unrecognized argument '-'" + string(fun) + ".");
        }
    }

    return run_benchmarks(filter_benchmarks(all_benchmarks, regexs));
}
