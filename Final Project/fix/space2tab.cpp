#include <iostream>
#include <fstream>
int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr<<"usage: "<<argv[0]<<" <input file> <output file>\n";
        return 1;
    }
    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cerr<<"unable to read file \""<<argv[1]<<"\"\n";
        return 1;
    }
    std::ofstream fout(argv[2]);
    if (!fout) {
        std::cerr<<"unable to write to \""<<argv[2]<<"\"\n";
        return 1;
    }
    char ch;
    int indent = 1;
    int sp = 0;
    while(fin.get(ch)) {
        if (indent) {
            if (ch == ' ') sp++;
            else { fout.put(ch); sp = 0; }
            if (sp == 4) {
                fout.put('\t'); sp = 0;
            }
            if (ch != ' ' && ch != '\t') indent = 0;
        }
        else {
            fout.put(ch);
            if (ch == '\n') indent = 1;
        }
    }
    return 0;
}
