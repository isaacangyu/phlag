#define DRIVER_VERSION "1_sliding_global_summary_clean_filename"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

using namespace std;

struct DataType16 {
    typedef unsigned short FreqType;
    typedef double EqFreqType;
    typedef double ScoreType;
    typedef long long CounterType;
};

template<typename DataType> 
class DStarQuadrupartitionScorer {
public:
    typedef typename DataType::FreqType FreqType;
    typedef typename DataType::EqFreqType EqFreqType;
    typedef typename DataType::ScoreType ScoreType;
    typedef typename DataType::CounterType CounterType;

    struct Block {
        array<vector<FreqType>, 4> cnt0, cnt1, cnt2, cnt3;
        vector<array<EqFreqType, 4> > pi;
        int windowSize;
    };
    
private:
    inline static CounterType quadXXYY(CounterType x0, CounterType x1, CounterType x2, CounterType x3, CounterType y0, CounterType y1, CounterType y2, CounterType y3) {
        return x0 * x1 * y2 * y3 + y0 * y1 * x2 * x3;
    }

    static ScoreType scoreSite(int pos, const array<vector<FreqType>, 4> &cnt0, const array<vector<FreqType>, 4> &cnt1,
            const array<vector<FreqType>, 4> &cnt2, const array<vector<FreqType>, 4> &cnt3, const array<EqFreqType, 4> &pi) {
        const EqFreqType A = pi[0], C = pi[1], G = pi[2], T = pi[3];
        const EqFreqType R = A + G, Y = C + T, R2 = A * A + G * G, Y2 = C * C + T * T;
        const FreqType a0 = cnt0[0][pos], c0 = cnt0[1][pos], g0 = cnt0[2][pos], t0 = cnt0[3][pos], r0 = a0 + g0, y0 = c0 + t0;
        const FreqType a1 = cnt1[0][pos], c1 = cnt1[1][pos], g1 = cnt1[2][pos], t1 = cnt1[3][pos], r1 = a1 + g1, y1 = c1 + t1;
        const FreqType a2 = cnt2[0][pos], c2 = cnt2[1][pos], g2 = cnt2[2][pos], t2 = cnt2[3][pos], r2 = a2 + g2, y2 = c2 + t2;
        const FreqType a3 = cnt3[0][pos], c3 = cnt3[1][pos], g3 = cnt3[2][pos], t3 = cnt3[3][pos], r3 = a3 + g3, y3 = c3 + t3;

        const CounterType rryy = quadXXYY(r0, r1, r2, r3, y0, y1, y2, y3);

        const CounterType aayy = quadXXYY(a0, a1, a2, a3, y0, y1, y2, y3);
        const CounterType ggyy = quadXXYY(g0, g1, g2, g3, y0, y1, y2, y3);
        const CounterType rrcc = quadXXYY(r0, r1, r2, r3, c0, c1, c2, c3);
        const CounterType rrtt = quadXXYY(r0, r1, r2, r3, t0, t1, t2, t3);
        
        const CounterType aacc = quadXXYY(a0, a1, a2, a3, c0, c1, c2, c3);
        const CounterType aatt = quadXXYY(a0, a1, a2, a3, t0, t1, t2, t3);
        const CounterType ggcc = quadXXYY(g0, g1, g2, g3, c0, c1, c2, c3);
        const CounterType ggtt = quadXXYY(g0, g1, g2, g3, t0, t1, t2, t3);
        
        return rryy * R2 * Y2 - (aayy + ggyy) * (R * R) * Y2 - (rrcc + rrtt) * R2 * (Y * Y)
            + (aacc + aatt + ggcc + ggtt) * (R * R) * (Y * Y);
    }

public:
    static ScoreType scoreInterval(int start, int end, const array<vector<FreqType>, 4> &cnt0, const array<vector<FreqType>, 4> &cnt1,
            const array<vector<FreqType>, 4> &cnt2, const array<vector<FreqType>, 4> &cnt3, const array<EqFreqType, 4> &pi) {
        ScoreType res = 0;
        for (int i = start; i < end; i++) res += scoreSite(i, cnt0, cnt1, cnt2, cnt3, pi);
        return res;
    }
    
    static vector<ScoreType> dstar(int windowSize, const array<vector<FreqType>, 4> &cnt0, const array<vector<FreqType>, 4> &cnt1,
            const array<vector<FreqType>, 4> &cnt2, const array<vector<FreqType>, 4> &cnt3, const vector<array<EqFreqType, 4> > &pi) {
        vector<ScoreType> res;
        for (int i = 0; i < cnt0[0].size(); i += 1) { 
            int j = (i + windowSize < cnt0[0].size()) ? i + windowSize : cnt0[0].size();
            res.push_back(scoreInterval(i, j, cnt0, cnt1, cnt2, cnt3, pi[i / windowSize]));
        }
        return res;
    }
    
    static CounterType quartetCnt(int start, int end, const array<vector<FreqType>, 4> &cnt0, const array<vector<FreqType>, 4> &cnt1,
            const array<vector<FreqType>, 4> &cnt2, const array<vector<FreqType>, 4> &cnt3) {
        CounterType res = 0;
        for (int i = start; i < end; i++) {
            CounterType s0 = cnt0[0][i] + cnt0[1][i] + cnt0[2][i] + cnt0[3][i];
            CounterType s1 = cnt1[0][i] + cnt1[1][i] + cnt1[2][i] + cnt1[3][i];
            CounterType s2 = cnt2[0][i] + cnt2[1][i] + cnt2[2][i] + cnt2[3][i];
            CounterType s3 = cnt3[0][i] + cnt3[1][i] + cnt3[2][i] + cnt3[3][i];
            res += s0 * s1 * s2 * s3;
        }
        return res;
    }
    
    static vector<CounterType> dstarQuartetCnt(int windowSize, const array<vector<FreqType>, 4> &cnt0, const array<vector<FreqType>, 4> &cnt1,
            const array<vector<FreqType>, 4> &cnt2, const array<vector<FreqType>, 4> &cnt3) {
        vector<CounterType> res;
        for (int i = 0; i < cnt0[0].size(); i += windowSize) {
            int j = (i + windowSize < cnt0[0].size()) ? i + windowSize : cnt0[0].size();
            res.push_back(quartetCnt(i, j, cnt0, cnt1, cnt2, cnt3));
        }
        return res;
    }

    static Block parseFreqs(const array<vector<FreqType>, 4> &f1, const array<vector<FreqType>, 4> &f2, 
            const array<vector<FreqType>, 4> &f3, const array<vector<FreqType>, 4> &f4, int start, int end, int windowSize) {
        Block res;
        res.windowSize = windowSize;
        res.pi.resize((end - start + windowSize - 1) / windowSize);
        array<const array<vector<FreqType>, 4>*, 4> lst = {&f1, &f2, &f3, &f4};
        array<array<vector<FreqType>, 4>*, 4> cntlst = {&res.cnt0, &res.cnt1, &res.cnt2, &res.cnt3};
        for (int i = 0; i < 4; i++) {
            const array<vector<FreqType>, 4> &f = *(lst[i]);
            array<vector<FreqType>, 4> &cnt = *(cntlst[i]);
            for (int k = 0; k < 4; k++) {
                for (int j = start; j < end; j++) {
                    cnt[k].push_back(f[k][j]);
                    res.pi[(j - start) / windowSize][k] += f[k][j];
                }
            }
        }
        for (int i = 0; i < res.pi.size(); i++) {
            EqFreqType sum = res.pi[i][0] + res.pi[i][1] + res.pi[i][2] + res.pi[i][3];
            for (int k = 0; k < 4; k++) res.pi[i][k] = (sum == 0) ? 0.25 : res.pi[i][k] / sum;
        }
        return res;
    }

    static string multiind(string input, string mapping = "", int intervalSize = 1000000, int windowSize = 10000, bool header = true) {
        string name[4];
        unordered_map<string, int> name2id;
        unordered_map<string, int> partname2id;
        if (mapping != "") {
            ifstream fmap(mapping);
            string idname, partname;
            while(fmap >> idname) {
                fmap >> partname;
                if (!partname2id.count(partname)) {
                    if (partname != "-") {
                        name[partname2id.size()] = partname;
                        partname2id[partname] = partname2id.size();
                    }
                    else partname2id[partname] = -1;
                }
                name2id[idname] = partname2id[partname];
            }
        }
        ifstream fin(input);
        string line;
        int id = -1, pos = 0;
        array<array<vector<FreqType>, 4>, 4> freq;
        while (getline(fin, line)) {
            if (line[0] == '>') {
                if (!name2id.count(line.substr(1))) {
                    name[partname2id.size()] = line.substr(1);
                    partname2id[line.substr(1)] = partname2id.size();
                    name2id[line.substr(1)] = partname2id[line.substr(1)];
                }
                id = name2id[line.substr(1)];
                pos = 0;
            }
            else if (id != -1) {
                for (size_t j = 0; j < line.size(); j++) {
                    for (int k = 0; k < 4; k++) {
                        if (pos + j >= freq[id][k].size()) freq[id][k].push_back(0); 
                    }
                    freq[id][0][pos + j] += (line[j] == 'A' || line[j] == 'a');
                    freq[id][1][pos + j] += (line[j] == 'C' || line[j] == 'c');
                    freq[id][2][pos + j] += (line[j] == 'G' || line[j] == 'g');
                    freq[id][3][pos + j] += (line[j] == 'T' || line[j] == 't'); 
                }
                pos += line.size();
            }
        }
        
        // 1. Isolate clean filename string from parent path
        size_t last_slash = input.find_last_of("/\\");
        string clean_filename = (last_slash == string::npos) ? input : input.substr(last_slash + 1);

        // 2. Truncate trailing .fa or .fasta extensions cleanly
        size_t last_dot = clean_filename.find_last_of(".");
        if (last_dot != string::npos) {
            clean_filename = clean_filename.substr(0, last_dot);
        }

        ostringstream sliding_file_out;
        if (header) {
            sliding_file_out << "pos\tavg*ABBA\tavg*BABA\tavg*AABB\tsliding_D*\tQuartetCnt\n";
        }
        
        vector<double> all_sliding_avgs1;
        vector<double> all_sliding_avgs2;
        vector<double> all_sliding_avgs3;
        vector<double> all_sliding_dstar; 

        double global_qcnt = 0;

        // Overlapping loop sequence step mapping (pos += 1)
        for (size_t pos_idx = 0; pos_idx < freq[0][0].size(); pos_idx += 1) {
            int end = (pos_idx + intervalSize < freq[0][0].size()) ? pos_idx + intervalSize : freq[0][0].size();
            Block data = parseFreqs(freq[0], freq[1], freq[2], freq[3], pos_idx, end, windowSize);
            vector<ScoreType> topology1 = dstar(windowSize, data.cnt0, data.cnt3, data.cnt1, data.cnt2, data.pi);
            vector<ScoreType> topology2 = dstar(windowSize, data.cnt1, data.cnt3, data.cnt0, data.cnt2, data.pi);
            vector<ScoreType> topology3 = dstar(windowSize, data.cnt2, data.cnt3, data.cnt0, data.cnt1, data.pi);
            vector<CounterType> quartetCnt = dstarQuartetCnt(windowSize, data.cnt0, data.cnt1, data.cnt2, data.cnt3);
            
            double sum1 = 0, sum2 = 0, sum3 = 0, qcnt = 0;
            size_t n = topology1.size();

            for (size_t i = 0; i < n; i++) {
                sum1 += topology1[i];
                sum2 += topology2[i];
                sum3 += topology3[i];
                qcnt += quartetCnt[i];
            }
            
            double avg1 = (n > 0) ? sum1 / n : 0;
            double avg2 = (n > 0) ? sum2 / n : 0;
            double avg3 = (n > 0) ? sum3 / n : 0;
            
            double sliding_d = 0;
            if ((sum1 + sum2 + sum3) != 0) {
                sliding_d = (sum1 - sum2) / (sum1 + sum2 + sum3);
            }

            // Write matrix logs without leading file attributes
            sliding_file_out << pos_idx << "\t" << avg1 << "\t" << avg2 << "\t" << avg3 << "\t" << sliding_d << "\t" << qcnt << "\n";
            
            all_sliding_avgs1.push_back(avg1);
            all_sliding_avgs2.push_back(avg2);
            all_sliding_avgs3.push_back(avg3);
            all_sliding_dstar.push_back(sliding_d); 

            global_qcnt += qcnt;
        }
        
        // Save complete sliding matrix table output to filename_sliding.tsv
        string data_save_target = clean_filename + "_sliding.tsv";
        ofstream disk_output(data_save_target);
        if (disk_output.is_open()) {
            disk_output << sliding_file_out.str();
            disk_output.close();
            cerr << "SUCCESS: High-density sliding window table created: " << data_save_target << endl;
        } else {
            cerr << "ERROR: Failed to open disk handle for writing path: " << data_save_target << endl;
        }

        // --- COMPUTE MACRO GLOBAL STATISTICS ---
        size_t total_intervals = all_sliding_avgs1.size();
        
        double global_avg1 = 0, global_avg2 = 0, global_avg3 = 0, dstar_avg = 0;
        double global_med1 = 0, global_med2 = 0, global_med3 = 0, dstar_med = 0;
        double global_std1 = 0, global_std2 = 0, global_std3 = 0, dstar_std = 0;

        if (total_intervals > 0) {
            double running_sum_avg1 = 0, running_sum_avg2 = 0, running_sum_avg3 = 0, dstar_sum = 0;
            for(size_t i = 0; i < total_intervals; ++i) {
                running_sum_avg1 += all_sliding_avgs1[i];
                running_sum_avg2 += all_sliding_avgs2[i];
                running_sum_avg3 += all_sliding_avgs3[i];
                dstar_sum        += all_sliding_dstar[i];
            }
            global_avg1 = running_sum_avg1 / total_intervals;
            global_avg2 = running_sum_avg2 / total_intervals;
            global_avg3 = running_sum_avg3 / total_intervals;
            dstar_avg   = dstar_sum / total_intervals;

            std::sort(all_sliding_avgs1.begin(), all_sliding_avgs1.end());
            std::sort(all_sliding_avgs2.begin(), all_sliding_avgs2.end());
            std::sort(all_sliding_avgs3.begin(), all_sliding_avgs3.end());
            std::sort(all_sliding_dstar.begin(), all_sliding_dstar.end());

            if (total_intervals % 2 != 0) {
                global_med1 = all_sliding_avgs1[total_intervals / 2];
                global_med2 = all_sliding_avgs2[total_intervals / 2];
                global_med3 = all_sliding_avgs3[total_intervals / 2];
                dstar_med   = all_sliding_dstar[total_intervals / 2];
            } else {
                size_t mid = total_intervals / 2;
                global_med1 = (all_sliding_avgs1[mid - 1] + all_sliding_avgs1[mid]) / 2.0;
                global_med2 = (all_sliding_avgs2[mid - 1] + all_sliding_avgs2[mid]) / 2.0;
                global_med3 = (all_sliding_avgs3[mid - 1] + all_sliding_avgs3[mid]) / 2.0;
                dstar_med   = (all_sliding_dstar[mid - 1] + all_sliding_dstar[mid]) / 2.0;
            }

            double var1 = 0, var2 = 0, var3 = 0, dstar_var = 0;
            for(size_t i = 0; i < total_intervals; ++i) {
                var1 += pow(all_sliding_avgs1[i] - global_avg1, 2);
                var2 += pow(all_sliding_avgs2[i] - global_avg2, 2);
                var3 += pow(all_sliding_avgs3[i] - global_avg3, 2);
                dstar_var += pow(all_sliding_dstar[i] - dstar_avg, 2);
            }
            double denom = (total_intervals > 1) ? total_intervals - 1 : 1;
            global_std1 = sqrt(var1 / denom);
            global_std2 = sqrt(var2 / denom);
            global_std3 = sqrt(var3 / denom);
            dstar_std   = sqrt(dstar_var / denom);
        }

        // Return strictly the consolidated global metrics summary table
        ostringstream summary_out;
        if (header) {
            summary_out << "metric\tavg*ABBA\tavg*BABA\tavg*AABB\tD*\n";
        }
        summary_out << "MEAN\t" << global_avg1 << "\t" << global_avg2 << "\t" << global_avg3 << "\t" << dstar_avg << "\n";
        summary_out << "MEDIAN\t" << global_med1 << "\t" << global_med2 << "\t" << global_med3 << "\t" << dstar_med << "\n";
        summary_out << "STDDEV\t" << global_std1 << "\t" << global_std2 << "\t" << global_std3 << "\t" << dstar_std << "\n";
        
        return summary_out.str();    
    }
};

const string HELP = R"V0G0N(D* Statistic Global Summary Tool
dstar FASTA_FILE [ MAPPING_FILE WINDOW_SIZE ]
)V0G0N";

int main(int argc, char *argv[])
{
    if (argc == 1 || argv[1][0] == '-') {
        cerr << HELP;
        return 0;
    }
    
    string fasta = argv[1];
    string mapping = (argc > 2) ? argv[2] : "-";
    if (mapping == "-") mapping = "";
    int window_size = (argc > 3) ? stoi(argv[3]) : 10000;
    int step_size = (argc > 4) ? stoi(argv[4]) : window_size;
    
    string global_summary = DStarQuadrupartitionScorer<DataType16>::multiind(fasta, mapping, window_size, step_size);

    // Prints the clean metrics text table out to console stdout
    cout << global_summary;
    
    return 0;
}