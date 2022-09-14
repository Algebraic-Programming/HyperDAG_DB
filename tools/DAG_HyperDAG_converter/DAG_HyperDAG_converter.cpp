/*
Copyright 2022 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Pal Andras Papp
*/


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <algorithm>

using namespace std;

bool DebugMode = false;


// DAG DATA STRUCTURE

struct DAG
{
    int n;
    vector< vector <int> > In, Out;
    string desc;

    //weight parameters
    vector<int> workW, commW;

    DAG(int N)
    {
        n=N;
        In.resize(n);
        Out.resize(n);
    }

    void addEdge(int v1, int v2, bool noPrint = false)
    {
        if(v2>=n)
            cout<<"Error: node index out of range."<<endl;

        In[v2].push_back(v1);
        Out[v1].push_back(v2);

        if(DebugMode && !noPrint)
            cout<<"Edge ("<<v1<<","<<v2<<")\n";
    }

    void Resize(int N)
    {
        n=N;
        In.clear();
        In.resize(n);
        Out.clear();
        Out.resize(n);
        workW.clear();
        workW.resize(n, 1);
        commW.clear();
        commW.resize(n, 1);
    }

    bool read(string filename, bool isDAG=true)
    {
        ifstream infile(filename);
        if(!infile.is_open())
        {
            cout<<"Unable to find/open input schedule file.\n";
            return false;
        }

        string line;
        getline(infile, line);
        while(!infile.eof() && line.at(0)=='%')
        {
           desc+=(desc.empty()? "" : "\n")+line;
           getline(infile, line);
        }

        int edges, N, pins=1;

        if(isDAG)
            sscanf(line.c_str(), "%d %d", &N, &edges);
        else
            sscanf(line.c_str(), "%d %d %d", &edges, &N, &pins);

        if(N<=0 || edges<=0 || pins<=0)
        {
            cout<<"Incorrect input file format (number of nodes/edges/pins is not positive).\n";
            return false;
        }

        Resize(N);

        //read weights
        int NrOFWeightLines = isDAG ? 2*N : edges+N;
        map<int,int> hyperedgeWeights;
        for(int i=0; i<NrOFWeightLines; ++i)
        {
            if(infile.eof())
            {
                cout<<"Incorrect input file format (file terminated too early).\n";
                return false;
            }

            getline(infile, line);
            while(!infile.eof() && line.at(0)=='%')
                getline(infile, line);

            int id, weight;
            sscanf(line.c_str(), "%d %d", &id, &weight);

            if(weight<0 || id<0 || (isDAG && id>=N) || (!isDAG && i>=edges && id>=N) || (!isDAG && i<edges && id>=edges))
            {
                cout<<"Incorrect input file format (index out of range or negative weight).\n";
                return false;
            }

            if(isDAG)
            {
                if(i<N)
                    workW[id]=weight;
                else
                    commW[id]=weight;
            }
            else
            {
                if(i<edges)
                    hyperedgeWeights[id]=weight;
                else
                    workW[id]=weight;
            }

        }

        // read edges/pins
        vector<int> edgeSource(edges, -1);
        int NrOfEdgeLines = isDAG ? edges : pins;
        for(int i=0; i<NrOfEdgeLines; ++i)
        {
            if(infile.eof())
            {
                cout<<"Incorrect input file format (file terminated too early).\n";
                return false;
            }
            getline(infile, line);
            while(!infile.eof() && line.at(0)=='%')
                getline(infile, line);

            int first, second;
            sscanf(line.c_str(), "%d %d", &first, &second);

            if(isDAG)
            {
                // interpret line as directed edge
                if(first<0 || second<0 || first>=N || second>=N)
                {
                    cout<<"Incorrect input file format (index out of range).\n";
                    return false;
                }
                addEdge(first, second);
            }
            else
            {
                // interpret line as pin
                if(first<0 || second<0 || first>=edges || second>=N)
                {
                    cout<<"Incorrect input file format (index out of range).\n";
                    return false;
                }
                if(edgeSource[first]==-1)
                {
                    edgeSource[first]=second;
                    commW[second] = hyperedgeWeights[first];
                }
                else
                    addEdge(edgeSource[first], second);
            }
        }

        ReOrderDAGVectors();

        infile.close();
        return true;
    }


    // Prints the DAG or hyperDAG corresponding to the input into a file
    void print(string filename, bool isDAG=false)
    {
        int sinks=0, pins=0, edges=0;
        for(int i=0; i<n; ++i)
        {
            edges+=Out[i].size();
            if(Out[i].size()>0)
                pins+=1+Out[i].size();
            else
                ++sinks;
        }

        ofstream outfile;
        outfile.open (filename);
        if(!desc.empty())
            outfile << desc <<"\n";

        if(isDAG)
            outfile << n << " " << edges <<endl;
        else
            outfile << n - sinks << " "<< n << " " << pins <<endl;



        if(isDAG) // DAG printing
        {
            //print node (work) weights
            for(int i=0; i<n; ++i)
                outfile <<i << " " << workW[i] <<"\n";
            //print node (communication) weights
            for(int i=0; i<n; ++i)
                outfile <<i << " " << commW[i] <<"\n";

            //print directed edges
            for(int i=0; i<n; ++i)
                for(int succ : Out[i])
                    outfile << i << " "<< succ <<"\n";

        }
        else // hyperDAG printing
        {
            //print hyperedge (communication) weights
            int edgeIndex = 0;
            for(int i=0; i<n; ++i)
                if(Out[i].size()>0)
                {
                    outfile << edgeIndex<<" "<<commW[i] <<"\n";
                    ++edgeIndex;
                }

            //print node (work) weights
            for(int i=0; i<n; ++i)
                outfile <<i << " " << workW[i] <<"\n";

            //print pins
            edgeIndex = 0;
            for(int i=0; i<n; ++i)
                if(Out[i].size()>0)
                {
                    outfile << edgeIndex << " "<< i <<"\n";
                    for(int j=0; j<Out[i].size(); ++j)
                        outfile << edgeIndex << " "<< Out[i][j] <<"\n";

                    ++edgeIndex;
                }
        }



        outfile.close();
     }

     void ReOrderDAGVectors()
     {
         for(int i=0; i<n; ++i)
         {
             sort(In[i].begin(), In[i].end());
             sort(Out[i].begin(), Out[i].end());
         }
     }
};


int main(int argc, char* argv[])
{
    string infile, outfile;
    bool toDAG = false, toHyperDAG = false;

    // PROCESS COMMAND LINE ARGUMENTS
    for (int i = 1; i < argc; ++i)
    {
        // Check parameters that require an argument afterwards
        if ((string(argv[i])=="-input" || string(argv[i])=="-output") && i + 1 >= argc)
        {
            cerr << "Parameter error: no parameter value after the \""<<string(argv[i])<<"\" option." << endl;
            return 1;
        }

        if (string(argv[i]) == "-input")
            infile = argv[++i];

        else if (string(argv[i]) == "-output")
            outfile = argv[++i];

        else if (string(argv[i]) == "-DAGtoHyperDAG")
            toHyperDAG = true;

        else if (string(argv[i]) == "-HyperDAGtoDAG")
            toDAG = true;

        else if (string(argv[i]) == "-debugMode")
            DebugMode = true;

        else
        {
            cerr << "Parameter error: unknown parameter/option "<< string(argv[i]) << endl;
            return 1;
        }
    }

    // check parameters
    if(infile.empty())
    {
        cerr << "Parameter error: no input file name specified." << endl;
        return 1;
    }
    if(toDAG == toHyperDAG)
    {
        cerr << "Parameter error: you should use exactly one of the \"-DAGtoHyperDAG\" and \"-HyperDAGtoDAG\" parameters." << endl;
        return 1;
    }
    if(outfile.empty())
    {
        outfile="outfile.txt";
        if(DebugMode)
            cout << "Output file not specified; using default output filename (outfile.txt)." << endl;
    }

    // Reading and converting DAG
    DAG G(1);
    if(!G.read(infile, toHyperDAG))
    {
        cerr << "Error reading from input file; no conversion possible." << endl;
        return 1;
    }
    G.print(outfile, toDAG);

    return 0;
}
