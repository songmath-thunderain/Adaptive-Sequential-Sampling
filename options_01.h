/*
   Header file for options 0 and 1 from main function.
   ****description of options 0 and 1.*****
*/

using Eigen::VectorXf;

void solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf);
void solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option);
