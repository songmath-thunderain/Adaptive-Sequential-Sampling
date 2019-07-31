/*
	Holds modified functions for the main function that will use new classes.
*/
#include "Solution.h"

Solution::Solution() {


}

Solution::~Solution() {

}

void Solution::preprocessing(IloEnv& env, TSLP& prob)
{
	// Rearrange the second-stage constraints data and variable bound data s.t.
	// 1 - eliminate unnecessary constraints like <= IloInfinity
	// 2 - all constraints are <=, >= or = constraints, if both bounds exist, append to the end of the constraint list
	// 3 - all variables are >= are = constraints
	for (int j = 0; j < prob.firstvarlb.getSize(); ++j)
	{
		if (prob.firstvarlb[j] == -1e10)
			prob.firstvarlb[j] = -IloInfinity;
		if (prob.firstvarub[j] == 1e10)
			prob.firstvarub[j] = IloInfinity;
	}

	for (int j = 0; j < prob.secondvarlb.getSize(); ++j)
	{
		if (prob.secondvarlb[j] == -1e10)
			prob.secondvarlb[j] = -IloInfinity;
		if (prob.secondvarub[j] == 1e10)
			prob.secondvarub[j] = IloInfinity;
	}

	// Construct CoefMatXf:
	prob.CoefMatXf = MatrixXf(prob.nbSecRows, prob.nbFirstVars);
	prob.CoefMatXf.setConstant(0);
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind < prob.nbFirstVars)
				prob.CoefMatXf(i, ind) = prob.CoefMat[i][j];
		}
	}
}


void Solution::solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf)
{
	// Benders: single cut
	double starttime = clock.getTime();
	bool feas_flag = 1;

	// Construct Master Problem
	Masterproblem master(env, prob, stat, clock, samples);
	// Add first-stage constraints and objective
	master.define_lp_model();

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subproblem subp(env, prob);

	subp.construct_second_opt(env, prob);
	subp.construct_second_feas(env, prob);

	int nbScens = samples.size();

	while ((stat.feasobjval - stat.relaxobjval) * 1.0 / (fabs(stat.feasobjval) + 1e-10) > 1e-6 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		IloNumArray xvals(env);
		double thetaval;
		double lasttime = clock.getTime();
		master.getCplex().solve();
		cout << "master status = " << master.getCplex().getStatus() << endl;
		if (master.getCplex().getStatus() == IloAlgorithm::Unbounded)
		{
			IloEnv meanenv;
			stat.relaxobjval = solve_mean_value_model(prob, meanenv, xvals, samples);
			meanenv.end();
		}
		else
		{
			stat.relaxobjval = master.getCplex().getObjValue();
			master.getCplex().getValues(xvals, master.getX());
			thetaval = master.getCplex().getValue(master.getTheta());
		}
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterateXf(j) = xvals[j];
		stat.mastertime += clock.getTime() - lasttime;
		double feasbound = 0.0;
		IloExpr lhsaggr(env);
		double lhsaggrval = 0;
		lhsaggr += master.getTheta();
		lhsaggrval += thetaval;
		double rhsaggr = 0;
		lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each scenario
			IloNumArray duals(env);
			bool feasflag;
			double subobjval = subp.solve(prob, xvals, duals, samples[k], feasflag, subp.calculate_bd(prob, xvals, samples[k]));
			

			VectorXf dualvec(prob.nbSecRows + prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows + prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = 0;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggrval += xvals[j] * opt_cut_coef[j] * 1.0 / nbScens;
						lhsaggr += master.getX()[j] * opt_cut_coef[j] * 1.0 / nbScens;
						sum_xvals += opt_cut_coef[j] * xvals[j];
					}
				}
				double rhssub = subobjval + sum_xvals;
				rhsaggr += rhssub * 1.0 / nbScens;
				feasbound += subobjval;
			}
			else
			{
				cout << "infeasible!" << endl;
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
						lhssub += master.getX()[j] * feas_cut_coef[j];
				}
				double rhssub = sum_xvals + subobjval;
				stat.num_feas_cuts++;
				master.getModel().add(lhssub >= rhssub);
				lhssub.end();
			}
		}
		stat.subtime += clock.getTime() - lasttime;
		if (feas_flag == 1)
		{
			feasbound = feasbound * 1.0 / nbScens;
			for (int j = 0; j < prob.nbFirstVars; ++j)
				feasbound += xvals[j] * prob.objcoef[j];
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			master.getModel().add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		else
			cout << "Infeasible! Generating feasibility cuts!" << endl;
		lhsaggr.end();
		xvals.end();
		cout << "relaxobjval = " << stat.relaxobjval << endl;
		cout << "feasobjval = " << stat.feasobjval << endl;
		cout << "optimality gap = " << (stat.feasobjval - stat.relaxobjval) * 1.0 / (fabs(stat.feasobjval) + 1e-10) << endl;
		stat.solvetime = clock.getTime() - starttime;
		cout << "stat.solvetime = " << stat.solvetime << endl;
		if (stat.solvetime > 10800)
			break;
	}
}



void Solution::solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option)
{
	// Level: level method starts with the mean-value solution
	// option = 0: solving SAA for getting a candidate solution; option = 1: solving SAA for evaluating a given solution using CI
	// first solve a mean-value problem
	double accuracy;
	if (option == 0)
	{
		// For getting a candidate solution
		accuracy = 1e-6;
	}
	if (option == 1)
	{
		// For evaluating a given solution using CI
		accuracy = 1e-4;
	}
	double starttime = clock.getTime();

	IloEnv meanenv;
	IloNumArray meanxvals(meanenv);
	double meanobj = solve_mean_value_model(prob, meanenv, meanxvals, samples);
	vector<double> xiterate(prob.nbFirstVars);
	// Assign xiterate to meanx
	for (int j = 0; j < prob.nbFirstVars; ++j)
		xiterate[j] = meanxvals[j];
	bool feas_flag = 1;
	meanxvals.end();
	meanenv.end();

	// Create and define level and quadratic Master Problems
	IloEnv lenv;
	Masterproblem master(env, prob, stat, clock, samples, lenv);
	master.define_lp_model();
	master.define_qp_model();

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subproblem subp(env, prob);
	subp.construct_second_opt(env, prob);
	subp.construct_second_feas(env, prob);

	int nbScens = samples.size();

	// This part needs to be changed into Level method iteration
	while ((stat.feasobjval - stat.relaxobjval) * 1.0 / (fabs(stat.feasobjval) + 1e-10) > accuracy || feas_flag == 0)
	{
		feas_flag = 1;
		IloEnv env2;
		stat.iter++;
		cout << "stat.iter = " << stat.iter << ", stat.feasobjval = " << stat.feasobjval << ", stat.relaxobjval = " << stat.relaxobjval << endl;
		IloNumArray xiteratevals(env2, prob.nbFirstVars);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			xiterateXf(j) = xiterate[j];
			xiteratevals[j] = xiterate[j];
		}
		double feasbound = 0.0;
		for (int j = 0; j < prob.nbFirstVars; ++j)
			feasbound += xiterate[j] * prob.objcoef[j];
		IloExpr lhsaggr(env);
		lhsaggr += master.getTheta();
		IloExpr llhsaggr(lenv);
		llhsaggr += master.getLtheta();
		double rhsaggr = 0;
		double lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each scenario
			IloNumArray duals(env2);
			bool feasflag;
			double subobjval = subp.solve(prob, xiteratevals, duals, samples[k], feasflag, subp.calculate_bd(prob, xiteratevals, samples[k]));
			VectorXf dualvec(prob.nbSecRows + prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows + prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = opt_cut_coef.dot(xiterateXf);
				double rhssub = subobjval + sum_xvals;
				feasbound += subobjval * 1.0 / nbScens;
				rhsaggr += rhssub * 1.0 / nbScens;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggr += master.getX()[j] * opt_cut_coef[j] * 1.0 / nbScens;
						llhsaggr += master.getLx()[j] * opt_cut_coef[j] * 1.0 / nbScens;
					}
				}

			}
			else
			{
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				IloExpr llhssub(lenv);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
					{
						llhssub += master.getLx()[j] * feas_cut_coef[j];
						lhssub += master.getX()[j] * feas_cut_coef[j];
					}
				}
				double rhssub = sum_xvals + subobjval;
				stat.num_feas_cuts++;
				master.getModel().add(lhssub >= rhssub);
				master.getLevelmodel().add(llhssub >= rhssub);
				lhssub.end();
				llhssub.end();
			}
		}
		xiteratevals.end();
		stat.subtime += clock.getTime() - lasttime;
		if (feas_flag == 1)
		{
			//cout << "feasbound = " << feasbound << ", feasobjval = " << stat.feasobjval << endl;
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			master.getLevelmodel().add(llhsaggr >= rhsaggr);
			master.getModel().add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		lhsaggr.end();
		llhsaggr.end();
		// Now solve the master, get a lower bound
		lasttime = clock.getTime();
		master.getCplex().solve();
		stat.relaxobjval = master.getCplex().getObjValue();
		stat.mastertime += clock.getTime() - lasttime;

		// Now solve the qp level problem
		// update the upper bound, (1-\lambda)F^{k+1}+\lambda F^*

		master.getRangeub().setUB(0.5 * stat.relaxobjval + 0.5 * stat.feasobjval);
		IloExpr objExpr(lenv);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			objExpr += master.getLx()[j] * master.getLx()[j];
			objExpr -= master.getLx()[j] * 2 * xiterate[j];
		}
		master.getLobj().setExpr(objExpr);
		objExpr.end();

		double startqptime = clock.getTime();
		master.getLevelcplex().solve();
		stat.qptime += clock.getTime() - startqptime;
		IloNumArray lxval(lenv);
		master.getLevelcplex().getValues(lxval, master.getLx());
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterate[j] = lxval[j];
		lxval.end();
		env2.end();
		//cout << "relaxobjval = " << stat.relaxobjval << endl;
		//cout << "feasobjval = " << stat.feasobjval << endl;
		//cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
		stat.solvetime = clock.getTime() - starttime;
		if (stat.solvetime > 10800)
			break;
	}

}

bool Solution::solve_partly_inexact_bundle(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option, bool initial, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int& nearOptimal, double remaintime)
{
	// option = 0: Just use the relative opt gap, solution mode: 1e-6; option = 1: Use the adaptive sampling type threshold: when opt gap is small relative to the sample error, option = 2: use relative opt gap, evaluation mode: 1e-4
	// initial = 0: a stability center is provided as xiterteXf from the previous iteration; initial = 1: the very first sampled problem solved
	// Unified coarse/fine oracle as a partly inexact oracle
	bool returnflag = 1;
	if (option == 1)
		cout << "begin solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "begin evaluation mode..." << endl;
	IloEnv env2;
	IloNumArray xvals(env2);
	double eval_threshold = 1e-4;
	double starttime = clock.getTime();
	double fmean;
	bool feas_flag = 0;
	IloNumArray stab_center(env, prob.nbFirstVars);
	vector< vector<double> > cutcoefs; // We may do some cut screening later
	vector<double> cutrhs;
	IloRangeArray cuts(env);
	double warmtemp = clock.getTime();

	// Create Partition object
	IloObjective QPobj;
	IloEnv meanenv;
	Masterproblem masterP(env, prob, stat, clock, samples);
	Subproblem subP(env, prob);
	subP.construct_second_opt(env, prob);
    subP.construct_second_feas(env, prob);
	Partition part_call(masterP, subP);
	if (initial == 0)
	{
		// One of the later iterations
		for (int j = 0; j < prob.nbFirstVars; ++j)
			stab_center[j] = xiterateXf(j);
		// We will also do some warm starts here using recorded dual information from dualCollection
		fmean = part_call.solve_warmstart(meanenv, prob, samples, stab_center, dualInfoCollection, cutcoefs, cutrhs, rhsvecs, xvals, clock, stat);

		// Add cuts
		for (int l = 0; l < cutcoefs.size(); ++l)
		{
			IloExpr lhs(env);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				lhs += part_call.masterProb.getX()[j] * (prob.objcoef[j] - cutcoefs[l][j] * 1.0 / samples.size());
			// Just tempararily set an UB, will be updated in the inner loop any way
			IloRange range(env, -IloInfinity, lhs, 0);
			cuts.add(range);
			lhs.end();
		}
		// This is only temporary
		Component all;
		for (int i = 0; i < samples.size(); ++i)
			all.indices.push_back(samples[i]);
		part_call.partition.push_back(all);
	}
	else
	{
		// Solve the mean-value problem first
		fmean = solve_mean_value_model(prob, meanenv, xvals, samples);
		// Set stab_center to be 0 initially
		for (int j = 0; j < prob.nbFirstVars; ++j)
			stab_center[j] = 0;
		// Initial partition: everybody is together, i.e., the mean-value problem
		Component all;
		for (int i = 0; i < samples.size(); ++i)
			all.indices.push_back(samples[i]);
		part_call.partition.push_back(all);
	}
	stat.warmstarttime = clock.getTime() - warmtemp;
	meanenv.end();
	IloRangeArray center_cons(env);
	//setup_bundle_QP(env, prob, part_call.masterProb.getCplex(), part_call.masterProb.getModel(), part_call.masterProb.getX(), stab_center, QPobj, cuts, center_cons);
	part_call.masterProb.setup_bundle_QP(stab_center, QPobj, cuts, center_cons);
	bool firstloop = 1;
	double descent_target;
	double opt_gap;
	double samplingError = 0; // samplingError will be updated whenever there is a chance to loop through all scenarios
	bool loopflag = 1;
	while (loopflag == 1 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		stat.partitionsize += part_call.partition.size();
		double feasboundscen = 0.0;
		VectorXf cutcoefscen(prob.nbFirstVars);
		cutcoefscen.setZero();
		VectorXf aggrCoarseCut(prob.nbFirstVars);
		vector<VectorXf> partcoef(part_call.partition.size());
		vector<double> partrhs(part_call.partition.size());
		double coarseLB, coarseCutRhs;
		vector<double> scenObjs; // record scenario objectives
		if (firstloop == 1)
			stat.relaxobjval = fmean;
		else
		{
			double lasttime = clock.getTime();
			coarseLB = part_call.coarse_oracle(env, prob, xvals, feasboundscen, cutcoefscen, stat, center_cons, stab_center, cuts, cutrhs, aggrCoarseCut, coarseCutRhs, partcoef, partrhs, starttime, clock, scenObjs, samples, dualInfoCollection, rhsvecs, option);
			stat.solvetime = clock.getTime() - starttime;
			if (stat.solvetime > remaintime)
			{
				returnflag = 0;
				break;
			}
			descent_target = prob.kappaf * stat.relaxobjval + (1 - prob.kappaf) * stat.feasobjval;
			// Here opt_gap is updated since stat.relaxobjval may have been updated during coarse_oracle()
			opt_gap = stat.feasobjval - stat.relaxobjval;
			if (option == 1 && samplingError > opt_gap * 10)
			{
				loopflag = 0;
				continue;
			}
			if (option == 0 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= prob.eps)
			{
				loopflag = 0;
				continue;
			}
			if (option == 2 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= eval_threshold)
			{
				//cout << "opt_gap = " << opt_gap << ", feasobjval = " << stat.feasobjval << endl;
				loopflag = 0;
				continue;
			}
			if (coarseLB > descent_target)
			{
				// Cannot achieve descent target, so just use coarse oracle
				IloExpr lhs(env);
				vector<double> tempcut(prob.nbFirstVars);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(aggrCoarseCut[j]) > 1e-7)
					{
						lhs += part_call.masterProb.getX()[j] * (prob.objcoef[j] - aggrCoarseCut[j] * 1.0 / samples.size());
						tempcut[j] = aggrCoarseCut[j];
					}
					else
					{
						lhs += part_call.masterProb.getX()[j] * prob.objcoef[j];
						tempcut[j] = 0;
					}
				}
				// Just tempararily set an UB, will be updated in the inner loop any way
				IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
				part_call.masterProb.getModel().add(range);
				cuts.add(range);
				cutcoefs.push_back(tempcut);
				cutrhs.push_back(coarseCutRhs);
				lhs.end();
				stat.num_opt_cuts++;
				if (loopflag != 0)
					continue;
			}
		}
		// Begin partition refinement
		vector<Component> new_partition;
		bool fullupdateflag = 0;
		if (firstloop == 1)
		{
			// just solve all the scenario subproblems
			feas_flag = part_call.refine_full(env, env2, prob, xvals, feasboundscen, cutcoefscen, new_partition, stat, clock, scenObjs, dualInfoCollection, rhsvecs, option);
			fullupdateflag = 1;
		}
		else
		{
			// Solve scenario subproblems component by component, stop when hopeless to achieve descent target
			feas_flag = part_call.refine_part(env, env2, prob, xvals, new_partition, stat, clock, partcoef, partrhs, descent_target, fullupdateflag, coarseLB, aggrCoarseCut, scenObjs, samples, dualInfoCollection, rhsvecs, option);
		}
		if (feas_flag == 1 && fullupdateflag == 1)
		{
			if (scenObjs.size() != samples.size())
			{
				cout << "scenObjs.size != samples.size" << endl;
				exit(0);
			}
			// A new candidate is available, update UB
			if (firstloop == 1)
			{
				double feasbound = 0.0;
				for (int j = 0; j < prob.nbFirstVars; ++j)
					feasbound += xvals[j] * prob.objcoef[j];
				feasbound += feasboundscen * 1.0 / samples.size();
				if (feasbound < stat.feasobjval - 1e-5)
					stat.feasobjval = feasbound;
				// update stablization center
				for (int j = 0; j < prob.nbFirstVars; ++j)
					stab_center[j] = xvals[j];
				computeSamplingError(samplingError, scenObjs);
			}
			else
			{
				if (coarseLB < stat.feasobjval - 1e-5)
					stat.feasobjval = coarseLB;
				// update stablization center
				if (coarseLB < descent_target)
				{
					computeSamplingError(samplingError, scenObjs);
					for (int j = 0; j < prob.nbFirstVars; ++j)
						stab_center[j] = xvals[j];
					// Begin bundle compression
					if (cuts.getSize() >= 100)
					{
						// Only do bundle compression if maintaining too many cuts in the bundle
						double maxval = -1e5;
						vector<double> cutvals(cuts.getSize());
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							double cutval = 0;
							for (int jj = 0; jj < prob.nbFirstVars; ++jj)
							{
								if (fabs(cutcoefs[j][jj]) > 1e-7 && fabs(stab_center[jj]) > 1e-7)
									cutval -= cutcoefs[j][jj] * stab_center[jj];
							}
							cutval = cutval * 1.0 / samples.size();
							cutval += cutrhs[j];
							cutvals[j] = cutval;
							if (cutval > maxval)
								maxval = cutval;
						}
						vector<bool> cutactive(cuts.getSize(), 0);
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							if (maxval - cutvals[j] < 1e-5)
								cutactive[j] = 1;
							else
							{
								// remove the cut
								part_call.masterProb.getModel().remove(cuts[j]);
								cuts[j].end();
							}
						}
						IloRangeArray newcuts(env);
						vector< vector<double> > newcutcoefs;
						vector<double> newcutrhs;
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							if (cutactive[j] == 1)
							{
								newcuts.add(cuts[j]);
								newcutcoefs.push_back(cutcoefs[j]);
								newcutrhs.push_back(cutrhs[j]);
							}
						}
						cutcoefs = newcutcoefs;
						cuts = newcuts;
						cutrhs = newcutrhs;
					}
				}
			}
		}
		opt_gap = stat.feasobjval - stat.relaxobjval;
		cout << "stat.iter = "<< stat.iter << ", relaxobjval = " << stat.relaxobjval << ", feasobjval = " << stat.feasobjval << ", optimality gap = " << opt_gap << ", samplingError = " << samplingError << endl;
		if (option == 0 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= 1e-6)
			loopflag = 0;
		if (option == 1 && samplingError > opt_gap * 10)
			loopflag = 0;
		if (option == 2 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= eval_threshold)
			loopflag = 0;
		//if (partition.size() == new_partition.size() && feas_flag == 1)
		//	loopflag = 0;
		//cout << "After updates, loopflag = " << loopflag << ", feas_flag = " << feas_flag << ", partition.size = " << partition.size() << ", new_partition.size = " << new_partition.size() << endl;
		if (loopflag == 1 && firstloop == 0)
		{
			part_call.partition = new_partition;
			// Need to add an aggregated cut, if feasible: maybe a fine oracle, or a mixed fine/coarse oracle
			if (feas_flag == 1)
			{
				if (firstloop == 1)
				{
					double lhsval = 0;
					vector<double> tempcut(prob.nbFirstVars);
					IloExpr lhs(env);
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(cutcoefscen[j]) > 1e-7)
						{
							lhs += part_call.masterProb.getX()[j] * (-cutcoefscen[j] * 1.0 / samples.size() + prob.objcoef[j]);
							lhsval += xvals[j] * cutcoefscen[j];
							tempcut[j] = cutcoefscen[j];
						}
						else
						{
							tempcut[j] = 0;
							if (fabs(prob.objcoef[j]) > 1e-7)
								lhs += part_call.masterProb.getX()[j] * prob.objcoef[j];
						}
					}
					double fineCutRhs = (feasboundscen + lhsval) * 1.0 / samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -fineCutRhs);
					part_call.masterProb.getModel().add(range);
					cuts.add(range);
					cutcoefs.push_back(tempcut);
					cutrhs.push_back(fineCutRhs);
					stat.num_opt_cuts++;
					lhs.end();
				}
				else
				{
					coarseCutRhs = coarseLB;
					for (int j = 0; j < prob.nbFirstVars; ++j)
						coarseCutRhs -= prob.objcoef[j] * xvals[j];
					IloExpr lhs(env);
					vector<double> tempcut(prob.nbFirstVars);
					double tempval = 0;
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(aggrCoarseCut[j]) > 1e-7)
						{
							lhs += part_call.masterProb.getX()[j] * (-aggrCoarseCut[j] * 1.0 / samples.size() + prob.objcoef[j]);
							tempval += xvals[j] * aggrCoarseCut[j];
							tempcut[j] = aggrCoarseCut[j];
						}
						else
						{
							tempcut[j] = 0;
							lhs += part_call.masterProb.getX()[j] * prob.objcoef[j];
						}
					}
					cutcoefs.push_back(tempcut);
					coarseCutRhs += tempval * 1.0 / samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
					part_call.masterProb.getModel().add(range);
					cuts.add(range);
					cutrhs.push_back(coarseCutRhs);
					stat.num_opt_cuts++;
					lhs.end();
				}
			}
		}
		firstloop = 0;
		stat.solvetime = clock.getTime() - starttime;
		if (stat.solvetime > remaintime)
		{
			returnflag = 0;
			break;
		}
	}
	if (stat.iter == 1) //nearly optimal after the very first iteration
	{
		if (nearOptimal == 0)
			nearOptimal = 1;
		else
			nearOptimal++;
	}
	else
	{
		if (nearOptimal < 3)
			nearOptimal = 0;
	}
	for (int jj = 0; jj < prob.nbFirstVars; ++jj)
		xiterateXf(jj) = stab_center[jj];
	stat.solvetime = clock.getTime() - starttime;
	stab_center.end();
	xvals.end();
	env2.end();
	cuts.end();
	center_cons.end();
	stat.finalpartitionsize = part_call.partition.size();
	if (option == 1)
		cout << "end solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "end evaluation mode..." << endl;
	return returnflag;
}


double Solution::solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples)
{
	IloModel meanmodel(meanenv);
	IloNumVarArray meanx(meanenv, prob.firstvarlb, prob.firstvarub);
	IloNumVarArray meany(meanenv, prob.nbSecVars, -IloInfinity, IloInfinity);
	// first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(meanenv);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += meanx[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
		IloRange range(meanenv, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		meanmodel.add(range);
		lhs.end();
	}
	int nbScens = samples.size();
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(meanenv);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs += meany[ind - prob.nbFirstVars] * prob.CoefMat[i][j];
			else
				lhs += meanx[ind] * prob.CoefMat[i][j];
		}
		// set constraint bounds
		double bd = 0;
		for (int k = 0; k < nbScens; ++k)
			bd += prob.secondconstrbd[samples[k] * prob.nbSecRows + i];
		bd = bd * 1.0 / nbScens;
		IloRange range;
		if (prob.secondconstrsense[i] == -1)
			range = IloRange(meanenv, bd, lhs, IloInfinity);
		if (prob.secondconstrsense[i] == 0)
			range = IloRange(meanenv, bd, lhs, bd);
		if (prob.secondconstrsense[i] == 1)
			range = IloRange(meanenv, -IloInfinity, lhs, bd);
		meanmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(meanenv);
		lhs += meany[i];
		IloRange range(meanenv, -IloInfinity, lhs, IloInfinity);
		if (prob.secondvarlb[i] != -IloInfinity)
		{
			double bd = 0;
			for (int k = 0; k < nbScens; ++k)
				bd += prob.secondvarlb[samples[k] * prob.nbSecVars + i];
			bd = bd * 1.0 / nbScens;
			range.setLB(bd);
		}
		if (prob.secondvarub[i] != IloInfinity)
		{
			double bd = 0;
			for (int k = 0; k < nbScens; ++k)
				bd += prob.secondvarub[samples[k] * prob.nbSecVars + i];
			bd = bd * 1.0 / nbScens;
			range.setUB(bd);
		}
		meanmodel.add(range);
		lhs.end();
	}
	IloExpr meanobj(meanenv);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		meanobj += meanx[i] * prob.objcoef[i];

	for (int i = 0; i < prob.objcoef.getSize(); ++i)
	{
		if (i >= prob.nbFirstVars)
			meanobj += meany[i - prob.nbFirstVars] * prob.objcoef[i];
	}
	meanmodel.add(IloMinimize(meanenv, meanobj));
	meanobj.end();
	IloCplex meancplex(meanmodel);
	meancplex.setParam(IloCplex::TiLim, 3600);
	meancplex.setParam(IloCplex::Threads, 1);
	meancplex.setParam(IloCplex::SimDisplay, 0);
	meancplex.setOut(meanenv.getNullStream());
	meancplex.solve();
	meancplex.getValues(meanxvals, meanx);
	double returnval = meancplex.getObjValue();
	meancplex.end();
	meanmodel.end();
	meanx.end();
	meany.end();
	return returnval;
}



void Solution::computeSamplingError(double& samplingError, const vector<double>& scenObjs)
{
	samplingError = 0;
	double sampleMean = 0;
	for (int k = 0; k < scenObjs.size(); ++k)
		sampleMean += scenObjs[k];
	sampleMean = sampleMean * 1.0 / scenObjs.size();
	for (int k = 0; k < scenObjs.size(); ++k)
		samplingError += pow(scenObjs[k] - sampleMean, 2);
	samplingError = sqrt(samplingError) * 1.0 / scenObjs.size();
	// \delta = 1e-3
	if (samplingError < 1e-3 * 1.0 / sqrt(scenObjs.size()))
		samplingError = 1e-3 * 1.0 / sqrt(scenObjs.size());
}



void Solution::setup_bundle_QP(IloEnv& env, const TSLP& prob, IloCplex& cplex, IloModel& model, IloNumVarArray& x, const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons)
{
	model = IloModel(env);
	// first stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			// Perform refinement
			lhs += x[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		model.add(range);
		lhs.end();
	}
	// Adding objective
	QPobj = IloMinimize(env);
	model.add(QPobj);
	IloExpr objExpr(env);
	IloNumVarArray y(env, prob.nbFirstVars, 0, IloInfinity);
	for (int j = 0; j < prob.nbFirstVars; ++j)
		objExpr += y[j];
	QPobj.setExpr(objExpr);
	objExpr.end();

	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		IloRange range(env, -IloInfinity, y[j] - x[j], IloInfinity);
		model.add(range);
		center_cons.add(range);
	}

	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		IloRange range(env, -IloInfinity, y[j] + x[j], IloInfinity);
		model.add(range);
		center_cons.add(range);
	}

	// Add cuts
	for (int l = 0; l < cuts.getSize(); ++l)
		model.add(cuts[l]);
	cplex = IloCplex(model);
	cplex.setParam(IloCplex::TiLim, 3600);
	cplex.setParam(IloCplex::Threads, 1);
	cplex.setParam(IloCplex::BarDisplay, 0);
	cplex.setParam(IloCplex::SimDisplay, 0);
	cplex.setOut(env.getNullStream());
	//cplex.setParam(IloCplex::EpOpt, 1e-4);
}



void Solution::solve_adaptive(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, int option, bool saaError)
{
	// option = 0: B&M (2011)
	// option = 1,2: B&P-L FSP/SSP
	// option = 3: use fixed rate schedule on increasing sample size, when BM fails to progress - when the new sampled problem is nearly optimal at the very first iteration for >= 3 times; also, once we switch to fixed rate schedule, we keep it in that way!
	// option = 4: use fixed rate schedule all the way
	// option = 5: use a simple heuristic "trust region" idea to adjust the sample size increasing rate

	// saaError = 0: traditional sequential sampling
	// saaError = 1: adaptive: solve SAA up to the sampling error

	double t_quant = 1.282;
	Sequence seq;
	sequentialSetup(seq, option);
	double epsilon; // epsilon is for stopping criterion of B&P-L
	bool iterflag = 1;
	int iter = 0;

	// Construct subproblems
	Subproblem subp(env,prob);
	subp.construct_second_opt(env, prob);
	subp.construct_second_feas(env, prob);

	// collection of dual multipliers
	vector<DualInfo> dualInfoCollection;
	VectorXf xiterateXf(prob.nbFirstVars);
	vector<VectorXf> rhsvecs;
	// Store all the rhs vectors
	for (int k = 0; k < prob.nbScens; ++k)
	{
		VectorXf rhsXf(prob.nbSecRows);
		for (int j = 0; j < prob.nbSecRows; ++j)
			rhsXf[j] = prob.secondconstrbd[j + k * prob.nbSecRows];
		rhsvecs.push_back(rhsXf);
	}
	int nearOptimal = 0; // keep track of # of consecutive times where the new sampled problem is nearly optimal at the very first iteration

	int nbIterEvalScens, nbIterSolScens;
	double starttime = clock.getTime();

	// We create some safeguard in case we do not finish by satisfying the stopping criterion in nMax steps
	double minGS = 1e8;
	double minGCI = 1e8;
	//double increaseRate = 0.1; // Starting from 0.1, lower bounded by 0.05
	while (iterflag == 1) // Outer loop
	{
		/* Setting up the sample size */
		if (iter == 0 || option == 0 || option == 1 || option == 2)
			nbIterEvalScens = seq.sampleSizes[iter];
		else
		{
			if (option == 4 || option == 5)
			{
				// option = 4: use a fixed rate all the way
				// option = 5: use a simple heuristic trust region idea to choose the sample size increasing rate
				nbIterEvalScens = int(nbIterEvalScens * (1 + prob.increaseRate));
				if (nbIterEvalScens < seq.sampleSizes[iter])
					nbIterEvalScens = seq.sampleSizes[iter];
			}
			if (option == 3)
			{
				if (nearOptimal >= 3)
				{
					nbIterEvalScens = int(nbIterEvalScens * (1 + prob.increaseRate));
					if (nbIterEvalScens < seq.sampleSizes[iter])
						nbIterEvalScens = seq.sampleSizes[iter];
				}
				else
				{
					if (seq.sampleSizes[iter] > nbIterEvalScens)
						nbIterEvalScens = seq.sampleSizes[iter];
				}
			}
		}
		nbIterSolScens = 2 * nbIterEvalScens;
		cout << "nbIterEvalScens = " << nbIterEvalScens << ", nbIterSolScens = " << nbIterSolScens << endl;

		/* Begin Solving sampled problems */
		vector<int> samplesForSol;
		bool evalFlag = 1;
		if (nbIterSolScens >= prob.nbScens)
		{
			// In case nbIterSolScens gets too big!
			for (int j = 0; j < prob.nbScens; ++j)
				samplesForSol.push_back(j);
			// No need to do probabilistic evaluation
			evalFlag = 0;
		}
		else
		{
			samplesForSol = vector<int>(nbIterSolScens);
			for (int j = 0; j < nbIterSolScens; ++j)
				samplesForSol[j] = rand() % prob.nbScens;
		}
		/* Construct Master problem in each iteration */
		IloModel mastermodel(env);
		IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
		IloNumVarArray theta(env, nbIterSolScens, 0, IloInfinity);
		// Adding first-stage constraints
		for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
		{
			IloExpr lhs(env);
			for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
				lhs += x[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
			IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
			mastermodel.add(range);
			lhs.end();
		}
		// Adding objective
		IloExpr obj(env);
		for (int i = 0; i < prob.nbFirstVars; ++i)
			obj += x[i] * prob.objcoef[i];
		for (int k = 0; k < nbIterSolScens; ++k)
			obj += theta[k] * (1.0 / nbIterSolScens);
		mastermodel.add(IloMinimize(env, obj));
		obj.end();

		IloCplex mastercplex(mastermodel);
		mastercplex.setParam(IloCplex::TiLim, 10800);
		mastercplex.setParam(IloCplex::Threads, 1);
		// Dual simplex
		mastercplex.setParam(IloCplex::RootAlg, 2);
		mastercplex.setParam(IloCplex::BarDisplay, 0);
		mastercplex.setParam(IloCplex::SimDisplay, 0);
		mastercplex.setOut(env.getNullStream());
		mastercplex.setParam(IloCplex::EpOpt, 1e-6);
		/* Finish construct Master problem in each iteration */

		// Adding all cutting planes from a collection of duals as constraints
		IloRangeArray cutcon(env);
		addInitialCuts(env, prob, mastermodel, x, theta, mastercplex, cutcon, samplesForSol, dualInfoCollection, xiterateXf, rhsvecs);
		bool newSampleFlag = 0;
		int innerIter = 0;
		double feasobjval = 1e10;
		double relaxobjval = -1e10;
		while (newSampleFlag == 0) // Inner loop
		{
			// After obtaining samples, solve the sampled problem, obtain xval, relaxation bound for the sampled problem
			innerIter++;
			if (innerIter > 1 || dualInfoCollection.size() * samplesForSol.size() < 10000)
				mastercplex.solve();
			relaxobjval = mastercplex.getObjValue();
			cout << "innerIter = " << innerIter << ", relaxobjval = " << relaxobjval << ", feasobjval = " << feasobjval << endl;;
			IloNumArray xvals(env);
			mastercplex.getValues(xvals, x);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				xiterateXf(j) = xvals[j];
			// solve the sampled subproblems, generate cuts, and then get an upper bound for the sampled problem
			double tempfeasobjval = 0;
			vector<double> scenobj;
			bool cutflag = 0;
			double sampleMean = 0;
			IloNumArray scenthetaval(env);
			mastercplex.getValues(scenthetaval, theta);
			double tempsubprobtime = clock.getTime();
			bool feas_flag = 1;
			for (int kk = 0; kk < nbIterSolScens; ++kk)
			{
				int k = samplesForSol[kk];
				IloNumArray duals(env);
				bool feasflag;
				double subobjval = subp.solve(prob, xvals, duals, k, feasflag, subp.calculate_bd(prob, xvals,k));
				VectorXf dualvec(prob.nbSecRows + prob.nbSecVars);
				for (int i = 0; i < prob.nbSecRows + prob.nbSecVars; ++i)
					dualvec(i) = duals[i];
				duals.end();
				if (feasflag == 1)
				{
					scenobj.push_back(subobjval);
					sampleMean += subobjval;
					if (subobjval > scenthetaval[kk] + max(1e-5, abs(scenthetaval[kk])) * 1e-5)
					{
						VectorXf opt_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
						double sum_xvals = 0;
						for (int j = 0; j < prob.nbFirstVars; ++j)
						{
							if (fabs(opt_cut_coef[j]) > 1e-7)
								sum_xvals += opt_cut_coef[j] * xvals[j];
						}
						double rhssub = subobjval + sum_xvals;
						// Need to add cuts here!
						IloExpr lhs(env);
						lhs += theta[kk];
						for (int j = 0; j < prob.nbFirstVars; ++j)
						{
							if (fabs(opt_cut_coef[j]) > 1e-7)
								lhs += x[j] * opt_cut_coef[j];
						}
						mastermodel.add(lhs >= rhssub);
						lhs.end();
						if (addToCollection(dualvec, dualInfoCollection) == true)
						{
							cutflag = 1;
							DualInfo dual;
							dual.dualvec = dualvec;
							dual.rhs = rhssub - dualvec.segment(0, prob.nbSecRows).transpose() * rhsvecs[k];
							dual.coefvec = opt_cut_coef;
							dualInfoCollection.push_back(dual);
						}
					}
					tempfeasobjval += subobjval;
				}
				else
				{
					cout << "; feas cut! Infeasibility = " << subobjval << endl;
					cutflag = 1;
					feas_flag = 0;
					VectorXf feas_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
					double sum_xvals = feas_cut_coef.dot(xiterateXf);
					IloExpr lhssub(env);
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(feas_cut_coef[j]) > 1e-7)
							lhssub += x[j] * feas_cut_coef[j];
					}
					double rhssub = sum_xvals + subobjval;
					stat.num_feas_cuts++;
					mastermodel.add(lhssub >= rhssub);
					lhssub.end();
					// adding as first-stage constraints to be used in the future
					IloIntArray tempInd(env);
					IloNumArray tempCoef(env);
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(feas_cut_coef[j]) > 1e-7)
						{
							tempInd.add(j);
							tempCoef.add(feas_cut_coef[j]);
						}
					}
					prob.firstconstrind.add(tempInd);
					prob.firstconstrcoef.add(tempCoef);
					prob.firstconstrlb.add(rhssub);
					prob.firstconstrub.add(IloInfinity);
					break;
				}
			}
			scenthetaval.end();
			if (feas_flag == 1)
			{
				// The following will only happen if no feasibility cuts are generated
				tempfeasobjval = tempfeasobjval * 1.0 / nbIterSolScens;
				for (int j = 0; j < prob.nbFirstVars; ++j)
					tempfeasobjval += xvals[j] * prob.objcoef[j];
				if (tempfeasobjval < feasobjval)
					feasobjval = tempfeasobjval;
				// Get the optimality gap with respective to the current sample
				double gap = feasobjval - relaxobjval;
				/* Now decide if we should solve a new sampled problem (with a larger sample) or continue solving the current sampled problem */
				if (saaError == 1 && evalFlag == 1)
				{
					// Get the sampling error
					double samplingError = 0;
					sampleMean = sampleMean * 1.0 / nbIterSolScens;
					for (int kk = 0; kk < nbIterSolScens; ++kk)
						samplingError += pow(scenobj[kk] - sampleMean, 2);
					samplingError = sqrt(samplingError) * 1.0 / nbIterSolScens;
					cout << "gap = " << gap << ", sample Error = " << samplingError << ", cutflag = " << cutflag << endl;
					if (samplingError > gap * 10 || cutflag == 0)
						newSampleFlag = 1;
				}
				else
				{
					cout << "gap = " << gap << ", cutflag = " << cutflag << endl;
					if (gap * 1.0 / (fabs(feasobjval) + 1e-10) <= 1e-6 || cutflag == 0)
						newSampleFlag = 1;
				}
			}
			if (newSampleFlag == 1)
			{
				if (evalFlag == 0)
				{
					iterflag = 0;
					stat.solvetime = clock.getTime() - starttime;
					stat.mainIter = iter;
					stat.finalSolExactObj = feasobjval;
					stat.gapThreshold = 0;
					stat.finalSampleSize = prob.nbScens;
					cout << "final objective = " << stat.finalSolExactObj << ", calculated by the full set of scenarios!" << endl;
				}
				else
				{
					// Finish inner iteration, need to do some SRP type testing
					if (innerIter == 1) //nearly optimal after the very first iteration
					{
						if (nearOptimal == 0)
							nearOptimal = 1;
						else
							nearOptimal++;
					}
					else
					{
						if (nearOptimal < 3)
							nearOptimal = 0;
					}
					// Now obtained xiterateXf, xvals, scenobj corresponding to \hat{x}_k
					if ((option == 1 || option == 2) && iter == 0)
					{
						// epsilon is for stopping criterion of B&P-L, just set it to be small enough relative to the initial UB obtained from the first iteration
						epsilon = prob.eps * fabs(feasobjval);
					}
					// Test stopping criterion: SRP type CI estimation
					double tempEvaltime = clock.getTime();
					double G = 0;
					double S = 0;
					vector<double> scenObjEval(nbIterEvalScens);
					vector<int> samplesForEval(nbIterEvalScens);
					for (int j = 0; j < nbIterEvalScens; ++j)
						samplesForEval[j] = rand() % prob.nbScens;
					STAT tempstat;
					tempstat.relaxobjval = -1e10;
					tempstat.feasobjval = 1e10;
					VectorXf xiterateXf2(prob.nbFirstVars);
					solve_level(env, prob, tempstat, clock, samplesForEval, xiterateXf2, 1);
					IloNumArray xvals2(env, prob.nbFirstVars);
					for (int j = 0; j < prob.nbFirstVars; ++j)
						xvals2[j] = xiterateXf2(j);
					// xiterateXf, xvals, scenObj correspondiing to \hat{x}_k, xiterateXf2, xvals2, scenObj2 corresponding to x^*_{n_k}
					SRP(env, prob, clock, nbIterEvalScens, samplesForEval, G, S, xvals, xvals2, scenObjEval);
					xvals2.end();
					stat.evaltime += (clock.getTime() - tempEvaltime);

					// Check if the stopping criterion is met
					if (option == 0)
					{
						// B&M (2011)
						cout << "G/S = " << G * 1.0 / S << ", G = " << G << ", S = " << S << endl;
						if (G * 1.0 / S < minGS)
							minGS = G * 1.0 / S;
						if (G <= seq.h2 * S + seq.eps2)
							iterflag = 0;
						else
							iter++;
					}
					if (option == 1 || option == 2)
					{
						cout << "CI width = " << G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens) << ", epsilon = " << epsilon << endl;
						// B&P-L: FSP/SSP
						if (G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens) < minGCI)
							minGCI = G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens);
						if (G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens) <= epsilon)
							iterflag = 0;
						else
						{
							iter++;
							if (option == 2)
							{
								// sample size for the next iteration will depend (adaptively) on the statistics of the current iteration
								double const_b = t_quant * S + 1;
								double const_c = nbIterEvalScens * G;
								double const_delta = const_b * const_b + 4 * epsilon * const_c;
								double const_v = (const_b + sqrt(const_delta)) * 1.0 / (2 * epsilon);
								seq.sampleSizes[iter] = int(const_v * const_v) + 1;
							}
						}
					}
					if (option == 3)
					{
						// Start with BM - if BM fails to progress - switch to fixed (exponential) rate schedule
						// Use either B&M (2011) or B&P-L criteria
						if (G * 1.0 / S < minGS)
							minGS = G * 1.0 / S;
						if (G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens) < minGCI)
							minGCI = G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens);
						if (G <= seq.h2 * S + seq.eps2 || G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens) <= epsilon)
							iterflag = 0;
						else
							iter++;
					}
					if (iter == seq.nMax)
						iterflag = 0;
					if (iterflag == 0)
					{
						cout << "# of iterations = " << iter << endl;
						stat.solvetime = clock.getTime() - starttime;

						// Now check if this estimation is correct by evaluating \hat{x}_k using the true distribution (all samples)
						finalEval(env, prob, xiterateXf, stat);
						stat.mainIter = iter;
						if (option == 0)
						{
							// B&M (2011)
							if (iter != seq.nMax)
								stat.gapThreshold = seq.h * S + seq.eps;
							else
							{
								stat.gapThreshold = (minGS + seq.h - seq.h2) * S + seq.eps;
							}
							cout << "optimality gap is less than " << stat.gapThreshold << " with prob >= 90%" << endl;
						}
						if (option == 1 || option == 2)
						{
							// B&P-L: FSP or SSP
							if (iter != seq.nMax)
								stat.gapThreshold = G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens);
							else
							{
								stat.gapThreshold = minGCI;
							}
							cout << "optimality gap is less than " << stat.gapThreshold << " with prob >= 90%" << endl;
						}
						if (option == 3 || option == 4 || option == 5)
						{
							// Use either B&M (2011) or B&P-L criteria
							if (iter != seq.nMax)
							{
								double gapThreshold = 1e8;
								if (G <= seq.h2 * S + seq.eps2)
									gapThreshold = seq.h * S + seq.eps;
								if (G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens) <= epsilon && G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens) < gapThreshold)
									gapThreshold = G + (t_quant * S + 1) * 1.0 / sqrt(nbIterEvalScens);
								stat.gapThreshold = gapThreshold;
							}
							else
							{
								if ((minGS + seq.h - seq.h2) * S + seq.eps < minGCI)
									stat.gapThreshold = (minGS + seq.h - seq.h2) * S + seq.eps;
								else
									stat.gapThreshold = minGCI;
							}
							cout << "optimality gap is less than " << stat.gapThreshold << " with prob >= 90%" << endl;
						}
						stat.finalSampleSize = nbIterSolScens;
					}
				}
			}
			xvals.end();
		}
		stat.iter += innerIter;
		cutcon.endElements();
		cutcon.end();
		mastercplex.end();
		mastermodel.end();
		x.end();
		theta.end();
	}

}



void Solution::finalEval(IloEnv& env, TSLP& prob, const VectorXf& xiterateXf, STAT& stat)
{
	Subproblem subprob(env, prob);
	IloNumArray finalxvals(env);
	double finalObj = 0;
	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		finalxvals.add(xiterateXf(j));
		finalObj += prob.objcoef[j] * xiterateXf(j);
	}
	double secStageObj = 0;
	for (int k = 0; k < prob.nbScens; ++k)
	{
		IloNumArray duals(env);
		bool feasflag = 0;
		double subobjval = subprob.solve(prob, finalxvals, duals, k, feasflag, subprob.calculate_bd(prob, finalxvals, k));
		if (feasflag == 0)
		{
			cout << "something is wrong!" << endl;
			exit(0);
		}
		duals.end();
		secStageObj += subobjval;
	}
	secStageObj = secStageObj * 1.0 / prob.nbScens;
	finalObj += secStageObj;
	cout << "final Obj = " << finalObj << endl;
	finalxvals.end();
	stat.finalSolExactObj = finalObj;
}

void Solution::SRP(IloEnv& env, TSLP& prob, IloTimer& clock, int nbIterEvalScens, const vector<int>& samplesForEval, double& G, double& S, const IloNumArray& xvals, const IloNumArray& xvals2, vector<double>& scenObjEval)
{
	Subproblem subprob;
	vector<double> scenObj2(nbIterEvalScens);
	for (int k = 0; k < nbIterEvalScens; ++k)
	{
		// solve subproblems for each scenario
		IloNumArray duals(env);
		bool feasflag;
		double subobjval = subprob.solve(prob, xvals, duals, samplesForEval[k], feasflag, subprob.calculate_bd(prob,xvals,samplesForEval[k]));
		if (feasflag == 0)
		{
			cout << "something is wrong!" << endl;
			exit(0);
		}
		IloNumArray duals2(env);
		double subobjval2 = subprob.solve(prob, xvals2, duals2, samplesForEval[k], feasflag, subprob.calculate_bd(prob, xvals2, samplesForEval[k]));
		if (feasflag == 0)
		{
			cout << "something is wrong!" << endl;
			exit(0);
		}
		duals.end();
		duals2.end();
		scenObjEval[k] = subobjval;
		scenObj2[k] = subobjval2;
	}

	// Use n_k (independent) samples to evaluate this candidate solution: compute G and S
	for (int k = 0; k < nbIterEvalScens; ++k)
		G += (scenObjEval[k] - scenObj2[k]);
	G = G * 1.0 / nbIterEvalScens;
	double deterGap = 0;
	for (int j = 0; j < prob.nbFirstVars; ++j)
		deterGap += prob.objcoef[j] * (xvals[j] - xvals2[j]);
	G += deterGap;
	for (int k = 0; k < nbIterEvalScens; ++k)
		S += pow(scenObjEval[k] - scenObj2[k] + deterGap - G, 2);
	S = S * 1.0 / (nbIterEvalScens - 1);
	S = sqrt(S);
}

void Solution::sequentialSetup(Sequence& seq, int option)
{
	// The sample size increasing schedule follows option 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP; option >= 3: geometric sequence, but lower bounded by B&M
	// At most 100 outer iterations, nMax and p are pairwise, used in B&M (2011)
	seq.nMax = 100; // int nMax = 500
	seq.p = 0.153; // double p = 0.104
	seq.h = 0.302;
	seq.h2 = 0.015;
	// eps, eps2 are for stopping criteria of B&M
	seq.eps = 2e-7;
	seq.eps2 = 1e-7;
	seq.sampleSizes = vector<int>(seq.nMax);
	//if (option == 0 || option >= 3)
	//	BMschedule(seq);
	seq.sampleSizes[0] = 50;
	int initSample = 50;
	if (option == 1 || option == 0)
	{
		// B&P-L: FSP, linear schedule
		for (int k = 0; k < seq.nMax; ++k)
			//seq.sampleSizes[k] = initSample+10*k;
			seq.sampleSizes[k] = initSample + 100 * k;
	}
	if (option == 2)
	{
		// B&P-L: SSP
		seq.sampleSizes[0] = initSample;
	}
}

void Solution::addInitialCuts(IloEnv& env, TSLP& prob, IloModel& mastermodel, const IloNumVarArray& x, const IloNumVarArray& theta, IloCplex& mastercplex, IloRangeArray& cutcon, const vector<int>& samplesForSol, const vector<DualInfo>& dualInfoCollection, const VectorXf& xiterateXf, const vector<VectorXf>& rhsvecs)
{
	// Given a collection of dual multipliers, construct an initial master problem (relaxation)
	if (dualInfoCollection.size() * samplesForSol.size() < 10000)
	{
		// Number of constraints is small enough to handle
		for (int l = 0; l < dualInfoCollection.size(); ++l)
		{
			for (int kk = 0; kk < samplesForSol.size(); ++kk)
			{
				int k = samplesForSol[kk];
				// assemble the cutcoef and cutrhs
				VectorXf opt_cut_coef = dualInfoCollection[l].coefvec;
				double opt_cut_rhs = dualInfoCollection[l].dualvec.segment(0, prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[l].rhs;
				IloExpr lhs(env);
				lhs += theta[kk];
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
						lhs += x[j] * opt_cut_coef[j];
				}
				IloRange range(env, opt_cut_rhs, lhs, IloInfinity);
				mastermodel.add(range);
				cutcon.add(range);
				lhs.end();
			}
		}
	}
	else
	{
		// Too many initial constraints, add them as cutting planes
		for (int kk = 0; kk < samplesForSol.size(); ++kk)
		{
			int k = samplesForSol[kk];
			// assemble the cutcoef and cutrhs
			double maxval = -1e8;
			int maxind = -1;
			for (int l = 0; l < dualInfoCollection.size(); ++l)
			{
				VectorXf opt_cut_coef = dualInfoCollection[l].coefvec;
				double opt_cut_rhs = dualInfoCollection[l].dualvec.segment(0, prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[l].rhs;
				double rhsval = opt_cut_rhs - opt_cut_coef.transpose() * xiterateXf;
				if (rhsval > maxval)
				{
					maxval = rhsval;
					maxind = l;
				}
			}
			IloExpr lhs(env);
			lhs += theta[kk];
			VectorXf init_cut_coef = dualInfoCollection[maxind].coefvec;
			for (int j = 0; j < prob.nbFirstVars; ++j)
			{
				if (fabs(init_cut_coef[j]) > 1e-7)
					lhs += x[j] * init_cut_coef[j];
			}
			IloRange range(env, dualInfoCollection[maxind].dualvec.segment(0, prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[maxind].rhs, lhs, IloInfinity);
			mastermodel.add(range);
			cutcon.add(range);
			lhs.end();
		}
		bool initialCutFlag = 1;
		while (initialCutFlag == 1)
		{
			initialCutFlag = 0;
			mastercplex.solve();
			IloNumArray xvals(env);
			IloNumArray thetavals(env);
			mastercplex.getValues(xvals, x);
			mastercplex.getValues(thetavals, theta);
			VectorXf tempxiterateXf(prob.nbFirstVars);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				tempxiterateXf(j) = xvals[j];
			for (int kk = 0; kk < samplesForSol.size(); ++kk)
			{
				int k = samplesForSol[kk];
				// assemble the cutcoef and cutrhs
				double maxval = -1e8;
				int maxind = -1;
				for (int l = 0; l < dualInfoCollection.size(); ++l)
				{
					VectorXf opt_cut_coef = dualInfoCollection[l].coefvec;
					double opt_cut_rhs = dualInfoCollection[l].dualvec.segment(0, prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[l].rhs;
					double rhsval = opt_cut_rhs - opt_cut_coef.transpose() * tempxiterateXf - thetavals[kk];
					if (rhsval > maxval)
					{
						maxval = rhsval;
						maxind = l;
					}
				}
				if (maxval > max(1e-5, abs(thetavals[kk])) * 1e-5)
				{
					initialCutFlag = 1;
					IloExpr lhs(env);
					lhs += theta[kk];
					VectorXf init_cut_coef = dualInfoCollection[maxind].coefvec;
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(init_cut_coef[j]) > 1e-7)
							lhs += x[j] * init_cut_coef[j];
					}
					IloRange range(env, dualInfoCollection[maxind].dualvec.segment(0, prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[maxind].rhs, lhs, IloInfinity);
					mastermodel.add(range);
					cutcon.add(range);
					lhs.end();
				}
			}
			xvals.end();
			thetavals.end();
		}
	}
}

bool Solution::addToCollection(const VectorXf& dualvec, vector<DualInfo>& dualInfoCollection)
{
	// try to add dualvec into dualCollection: if it is not parallel to any other vector in the collection
	bool flag = 1;
	for (int j = 0; j < dualInfoCollection.size(); ++j)
	{
		double par = dualvec.dot(dualInfoCollection[j].dualvec) * 1.0 / (dualvec.norm() * dualInfoCollection[j].dualvec.norm());
		if (par > 1 - 1e-3)
		{
			flag = 0;
			break;
		}
	}
	return flag;
}
