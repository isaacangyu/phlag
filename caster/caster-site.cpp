#define DRIVER_VERSION "6"

/* CHANGE LOG
 * 6: Fix corner case bug
 * 5: Final score normalization & Updating Eqfreq
 * 4: Updating Eqfreq calculation
 * 3: Updating input file parser
 * 2: Adding branch length functionality
 * 1: Modified the logic in parsing FASTA names 
 */

#include<iostream>
#include<fstream>
#include<unordered_map>
#include<unordered_set>
#include<map>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<random>
#include<thread>
#include<mutex>

#define BLOCK_BOOTSTRAP

//#define CUSTOMIZED_ANNOTATION_TERMINAL_LENGTH
#ifndef CUSTOMIZED_ANNOTATION_TERMINAL_LENGTH
#define CUSTOMIZED_ANNOTATION_LENGTH
#endif

//#define LARGE_DATA
#ifdef LARGE_DATA
typedef long double score_t;
typedef long long count_t;
#else
typedef double score_t;
typedef int count_t;
#endif

#include "sequence.hpp"
#include "algorithms.hpp"
#include "sequtils.hpp"

using namespace std;

int GENE_ID = 0;

struct Workflow {
    MetaAlgorithm meta;
    TripartitionInitializer& tripInit = meta.tripInit;

    vector<string>& names = meta.names;
    unordered_map<string, int>& name2id = meta.name2id;

    // Parallel to tripInit.genes -- which locus/chunk each gene came from, so
    // --branch-mapping's per-chunk scoring can group genes back by chunk
    // (formatGene() pushes exactly one gene per chunk here).
    vector<int> geneLocus, geneChunk;

    void addName(const string& name) {
        if (name2id.count(name) == 0) {
            name2id[name] = names.size();
            names.push_back(name);
        }
    }

    static int infoCount(const array<int, 4> &cnt){
        int result = 0;
        for (int j = 0; j < 4; j++) {
            if (cnt[j] >= 2) result++;
        }
        return result;
    }

    static array<int, 4> add(const array<int, 4> &a, const array<int, 4> &b){
        array<int, 4> result;
        for (int j = 0; j < 4; j++) {
            result[j] = a[j] + b[j];
        }
        return result;
    }

    static int sum(const array<int, 4> &cnt){
        int result = 0;
        for (int j = 0; j < 4; j++) {
            result += cnt[j];
        }
        return result;
    }

    void buildGeneSeq(TripartitionInitializer::Gene::Initializer &gene, const vector<int> &ind2species, size_t pos, size_t nSite, size_t offset, const array<double, 4> &eqfreq){
        int nInd = ind2species.size();
        for (int iInd = 0; iInd < nInd; iInd++) {
            size_t pSeq = pos + iInd * offset;
            gene.species2ind[ind2species[iInd]].push_back(iInd);
            gene.ind2seq[iInd] = pSeq;
        }
        for (int j = 0; j < 4; j++) {
            gene.pi[j] = eqfreq[j];
        }
    }

    void formatGene(const vector<int> &ind2species, size_t pos, size_t nSite, size_t offset, const array<double, 4> &eqfreq, int locusId, int chunkId) {
        int nInd = ind2species.size(), nSpecies = names.size(), nKernal = nSite, nRep = 0;
        TripartitionInitializer::Gene::Initializer gene(nInd, nSpecies, nSite, nKernal, nRep);
        buildGeneSeq(gene, ind2species, pos, nSite, offset, eqfreq);
        tripInit.genes.emplace_back(gene);
        geneLocus.push_back(locusId);
        geneChunk.push_back(chunkId);
    }

    void read(const string &file, const string &fileFormat, const string &seqFormat) {
        AlignmentParser AP(file, fileFormat, seqFormat), AP2(file, fileFormat, seqFormat);
        int locusId = 0;
        while (AP.nextAlignment()){
            AP2.nextAlignment();
            long long nSites = AP.getLength();
            long long nChunk = (nSites + ARG.getIntArg("chunk") - 1) / ARG.getIntArg("chunk");
            vector<vector<long long> > sites(nChunk);
            vector<array<double, 4> > eqfreq;
            long long offset = 0;
            {
                vector<array<unsigned short, 4> > freq;
                freq.resize(AP.getLength());
                while (AP.nextSeq()) {
                    addName(meta.mappedname(AP.getName()));
                    string seq = AP.getSeq();
                    for (size_t i = 0; i < seq.size(); i++) {
                        switch (seq[i]) {
                            case 'A': freq[i][0]++; break;
                            case 'C': freq[i][1]++; break;
                            case 'G': freq[i][2]++; break;
                            case 'T': freq[i][3]++; break;
                        }
                    }
                }
                for (long long i = 0; i < nChunk; i++) {
                    long long s = i * nSites / nChunk, t = (i + 1) * nSites / nChunk;
                    long long sumFreq[4] = {};
                    for (long long j = s; j < t; j++) {
                        for (int k = 0; k < 4; k++) {
                            sumFreq[k] += freq[j][k];
                        }
                        #ifdef CUSTOMIZED_ANNOTATION_TERMINAL_LENGTH
                        sites[i].push_back(j);
                        #else
                        if (freq[j][0] + freq[j][2] >= 2 && freq[j][1] + freq[j][3] >= 2) sites[i].push_back(j);
                        #endif
                    }
                    offset += sites[i].size();
                    double total = sumFreq[0] + sumFreq[1] + sumFreq[2] + sumFreq[3];
                    if (total > 0) eqfreq.push_back({ sumFreq[0] / total, sumFreq[1] / total, sumFreq[2] / total, sumFreq[3] / total });
					else eqfreq.push_back({ 0.25, 0.25, 0.25, 0.25 });
                }
            }
            {
                vector<int> ind2species;
                size_t pos = tripInit.seq.len();
                while (AP2.nextSeq()) {
                    ind2species.push_back(name2id[meta.mappedname(AP2.getName())]);
                    string seq = AP2.getSeq();
                    for (long long i = 0; i < nChunk; i++) {
                        for (long long j : sites[i]) tripInit.seq.append(seq[j]);
                    }
                }
                for (int i = 0; i < nChunk; i++) {
                    formatGene(ind2species, pos, sites[i].size(), offset, eqfreq[i], locusId, i);
                    pos += sites[i].size();
                }
            }
            locusId++;
        }
    }

    // --branch-mapping mode: skip tree search entirely and score one fixed,
    // caller-specified quartet branch per genomic chunk instead. mappingFile
    // is a 2-column TSV (taxon<TAB>group) -- the same format already used for
    // population mapping files elsewhere in this pipeline -- assigning every
    // taxon in the alignment to exactly one of 4 groups. Those 4 groups ARE
    // the branch: the two subtrees on each side of the edge being scored.
    // Mirrors caster-pair.cpp's scoreChunksForBranch.
    void scoreChunksForBranch(const string &mappingFile, const string &outFile){
        ifstream fmap(mappingFile);
        if (!fmap){
            cerr << "Error: could not open branch mapping file '" << mappingFile << "'\n";
            exit(1);
        }
        unordered_map<string, string> taxon2group;
        vector<string> groupOrder;
        unordered_map<string, int> group2idx;
        string taxon, group;
        while (fmap >> taxon >> group){
            taxon2group[taxon] = group;
            if (group2idx.count(group) == 0){
                group2idx[group] = (int) groupOrder.size();
                groupOrder.push_back(group);
            }
        }
        if (groupOrder.size() != 4){
            cerr << "Error: branch mapping file '" << mappingFile << "' has " << groupOrder.size()
                 << " distinct group(s) (" ;
            for (size_t i = 0; i < groupOrder.size(); i++) cerr << (i ? ", " : "") << groupOrder[i];
            cerr << "), need exactly 4 to define a branch.\n";
            exit(1);
        }

        vector<int> speciesGroup(names.size(), -1);
        for (size_t i = 0; i < names.size(); i++){
            auto it = taxon2group.find(names[i]);
            if (it == taxon2group.end()){
                cerr << "Error: species '" << names[i] << "' (present in the alignment) not found in branch mapping file '" << mappingFile << "'.\n";
                exit(1);
            }
            speciesGroup[i] = group2idx[it->second];
        }

        Quadripartition quad(tripInit);
        for (size_t i = 0; i < names.size(); i++){
            quad.update(speciesGroup[i], (int) i);
        }

        map<pair<int, int>, array<score_t, 3> > chunkScores;
        for (size_t a = 0; a < quad.genes.size(); a++){
            array<score_t, 3> s = quad.genes[a].scoreCnt();
            array<score_t, 3> &acc = chunkScores[make_pair(geneLocus[a], geneChunk[a])];
            acc[0] += s[0]; acc[1] += s[1]; acc[2] += s[2];
        }

        // Column names/positions mirror dstar's scores.tsv (pos, c*ABBA,
        // c*BABA, c*AABB) purely so downstream tooling (phlag/caster.py's
        // CasterPlotter) can render this exactly like a normal scores.tsv
        // scatter plot. c*ABBA/c*BABA/c*AABB have no ABBA/BABA/AABB
        // site-pattern meaning here -- they're just the 3 possible
        // resolutions of this branch's quartet, in a fixed order; the actual
        // group identities are logged below. Unlike caster-pair's
        // chunk_scores.tsv, no q1/q2/q3 columns: these raw sums are already
        // dstar-scale (~1e5-1e6, not caster-pair's ~1e11), so phlag's
        // read_caster_scores must not mistake this for a --pair file and
        // z-score it -- that detection keys off q1/q2/q3's presence alone.
        long long chunkSizeArg = ARG.getIntArg("chunk");
        cerr << "c*ABBA = " << groupOrder[0] << groupOrder[1] << "|" << groupOrder[2] << groupOrder[3]
             << ", c*BABA = " << groupOrder[0] << groupOrder[2] << "|" << groupOrder[1] << groupOrder[3]
             << ", c*AABB = " << groupOrder[0] << groupOrder[3] << "|" << groupOrder[1] << groupOrder[2] << "\n";

        ofstream fout(outFile);
        if (!fout){
            cerr << "Error: could not open '" << outFile << "' for writing.\n";
            exit(1);
        }
        const string &inputFile = ARG.getStringArg("input");
        fout << "file\tpos\tc*ABBA\tc*BABA\tc*AABB\n";
        for (auto &kv: chunkScores){
            long long pos = (long long) kv.first.second * chunkSizeArg;
            fout << inputFile << "\t" << pos << "\t"
                 << (double) kv.second[0] << "\t" << (double) kv.second[1] << "\t" << (double) kv.second[2] << "\n";
        }
        cerr << "Wrote per-chunk branch scores (" << chunkScores.size() << " chunks) to: " << outFile << "\n";
    }

    Workflow(int argc, char** argv){
        //string mappingFile;
        meta.initialize(argc, argv);
        if (ARG.getStringArg("root") != "") addName(ARG.getStringArg("root"));
        init();
    }

    void init(){
        tripInit.nThreads = meta.nThreads;

        string fileFormat = ARG.getStringArg("format");
        string seqFormat = (ARG.getIntArg("ambiguity")) ? "ambiguity" : "NA";
        if (fileFormat != "auto" && fileFormat != "fasta" && fileFormat != "phylip" && fileFormat != "list"){
            cerr << "Failed to parse format named '" << fileFormat << "'\n";
			exit(1);
        }
		read(ARG.getStringArg("input"), fileFormat, seqFormat);
        tripInit.nSpecies = names.size();
    }
};

int main(int argc, char** argv){
    ARG.setProgramName("caster-site", "Coalescence-aware Alignment-based Species Tree EstimatoR (Site)");
	ARG.addStringArg(0, "length", "SULength", "SULength: substitution-per-site unit; CULength: coalescent unit");
    ARG.addStringArg('f', "format", "auto", "Input file type, fasta: one fasta file for the whole alignment, list: a txt file containing a list of FASTA files, phylip: a phylip file for the whole alignment, auto (default): detect format automatically", true);
    ARG.addIntArg(0, "ambiguity", 0, "0 (default): ambiguity codes are treated as N, 1: ambiguity codes are treated as diploid unphased sites");
    ARG.addIntArg(0, "chunk", 10000, "The chunk size of each local region for parameter estimation");
    ARG.addStringArg(0, "branch-mapping", "", "Path to a 2-column TSV (taxon<TAB>group) assigning every taxon "
        "in the alignment to exactly one of 4 groups defining a fixed quartet branch (the same format as this "
        "pipeline's population mapping files, e.g. P1/P2/P3/Po). When given, skips the tree search entirely and "
        "instead scores that one branch's quartet topology per genomic chunk -- see --chunk-scores.");
    ARG.addStringArg(0, "chunk-scores", "chunk_scores.tsv", "Output TSV path for --branch-mapping's per-chunk quartet scores.");

    #ifdef CUSTOMIZED_ANNOTATION_TERMINAL_LENGTH
    cerr << "Warning: This version can be much slower and require much more memory than the regular version. You might want to compute the correct topology first and add branch lengths with this version on fix topology.\n";
    #endif

    Workflow WF(argc, argv);

    string branchMapping = ARG.getStringArg("branch-mapping");
    if (branchMapping != ""){
        WF.scoreChunksForBranch(branchMapping, ARG.getStringArg("chunk-scores"));
        return 0;
    }

	ARG.getStringArg("annotation") = "BootstrapSupport";

    LOG << "#Base: " << WF.meta.tripInit.seq.len() << endl;
    auto res = WF.meta.run();
    LOG << "Normalized score: " << (double) res.first / 4 << endl;
	return 0;
}
