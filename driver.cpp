bool solve_partly_inexact_bundle(IloEnv& env, TSLP& prob, Subprob& subp, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option, bool initial, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int& nearOptimal, double remaintime)
{
	// option = 0: Just use the relative opt gap, solution mode: 1e-6; option = 1: Use the adaptive sampling type threshold: when opt gap is small relative to the sample error, option = 2: use relative opt gap, evaluation mode: 1e-4
	// initial = 0: a stability center is provided as xiterteXf from the previous iteration; initial = 1: the very first sampled problem solved
	// Unified coarse/fine oracle as a partly inexact oracle


	bool returnflag = 1;
	if (option == 1)
		cout << "begin solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "begin evaluation mode..." << endl;
	double eval_threshold = 1e-4;
	double starttime = clock.getTime();
	IloEnv env2;
	IloNumArray xvals(env2);
	double fmean;
	vector<Component> partition;
	bool feas_flag = 0;
	IloCplex cplex;
	IloModel model;
	IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
	IloObjective QPobj;
	IloNumArray stab_center(env, prob.nbFirstVars);
	vector< vector<double> > cutcoefs; // We may do some cut screening later
	vector<double> cutrhs;
	IloRangeArray cuts(env);
	IloEnv meanenv;
	double warmtemp = clock.getTime();
	if (initial == 0)
	{
		// One of the later iterations
		for (int j = 0; j < prob.nbFirstVars; ++j)
			stab_center[j] = xiterateXf(j);
		// We will also do some warm starts here using recorded dual information from dualCollection
		fmean = solve_warmstart(meanenv, prob, samples, stab_center, dualInfoCollection, cutcoefs, cutrhs, partition, rhsvecs, xvals, clock, stat);
		// Add cuts
		for (int l = 0; l < cutcoefs.size(); ++l)
		{
			IloExpr lhs(env);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				lhs += x[j] * (prob.objcoef[j] - cutcoefs[l][j] * 1.0 / samples.size());
			// Just tempararily set an UB, will be updated in the inner loop any way
			IloRange range(env, -IloInfinity, lhs, 0);
			cuts.add(range);
			lhs.end();
		}
		// This is only temporary
		Component all;
		for (int i = 0; i < samples.size(); ++i)
			all.indices.push_back(samples[i]);
		partition.push_back(all);
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
		partition.push_back(all);
	}
	stat.warmstarttime = clock.getTime() - warmtemp;
	meanenv.end();
	IloRangeArray center_cons(env);
	setup_bundle_QP(env, prob, cplex, model, x, stab_center, QPobj, cuts, center_cons);
	bool firstloop = 1;
	double descent_target;
	double opt_gap;
	double samplingError = 0; // samplingError will be updated whenever there is a chance to loop through all scenarios
	bool loopflag = 1;
	while (loopflag == 1 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		//cout << "stat.iter = " << stat.iter << endl;
		stat.partitionsize += partition.size();
		double feasboundscen = 0.0;
		VectorXf cutcoefscen(prob.nbFirstVars);
		cutcoefscen.setZero();
		VectorXf aggrCoarseCut(prob.nbFirstVars);
		vector<VectorXf> partcoef(partition.size());
		vector<double> partrhs(partition.size());
		double coarseLB, coarseCutRhs;
		vector<double> scenObjs; // record scenario objectives
		if (firstloop == 1)
			stat.relaxobjval = fmean;
		else
		{
			double lasttime = clock.getTime();
			coarseLB = coarse_oracle(env, prob, subp, partition, xvals, feasboundscen, cutcoefscen, cplex, model, x, stat, center_cons, stab_center, cuts, cutrhs, aggrCoarseCut, coarseCutRhs, partcoef, partrhs, starttime, clock, scenObjs, samples, dualInfoCollection, rhsvecs, option);
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
						lhs += x[j] * (prob.objcoef[j] - aggrCoarseCut[j] * 1.0 / samples.size());
						tempcut[j] = aggrCoarseCut[j];
					}
					else
					{
						lhs += x[j] * prob.objcoef[j];
						tempcut[j] = 0;
					}
				}
				// Just tempararily set an UB, will be updated in the inner loop any way
				IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
				model.add(range);
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
			feas_flag = solve_scen_subprobs(env, env2, prob, subp, partition, xvals, feasboundscen, cutcoefscen, model, x, new_partition, stat, clock, scenObjs, dualInfoCollection, rhsvecs, option);
			fullupdateflag = 1;
		}
		else
		{
			// Solve scenario subproblems component by component, stop when hopeless to achieve descent target
			feas_flag = solve_scen_subprobs_target(env, env2, prob, subp, partition, xvals, model, x, new_partition, stat, clock, partcoef, partrhs, descent_target, fullupdateflag, coarseLB, aggrCoarseCut, scenObjs, samples, dualInfoCollection, rhsvecs, option);
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
								model.remove(cuts[j]);
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
		//cout << "stat.iter = "<< stat.iter << ", relaxobjval = " << stat.relaxobjval << ", feasobjval = " << stat.feasobjval << ", optimality gap = " << opt_gap << ", samplingError = " << samplingError << endl;
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
			partition = new_partition;
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
							lhs += x[j] * (-cutcoefscen[j] * 1.0 / samples.size() + prob.objcoef[j]);
							lhsval += xvals[j] * cutcoefscen[j];
							tempcut[j] = cutcoefscen[j];
						}
						else
						{
							tempcut[j] = 0;
							if (fabs(prob.objcoef[j]) > 1e-7)
								lhs += x[j] * prob.objcoef[j];
						}
					}
					double fineCutRhs = (feasboundscen + lhsval) * 1.0 / samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -fineCutRhs);
					model.add(range);
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
							lhs += x[j] * (-aggrCoarseCut[j] * 1.0 / samples.size() + prob.objcoef[j]);
							tempval += xvals[j] * aggrCoarseCut[j];
							tempcut[j] = aggrCoarseCut[j];
						}
						else
						{
							tempcut[j] = 0;
							lhs += x[j] * prob.objcoef[j];
						}
					}
					cutcoefs.push_back(tempcut);
					coarseCutRhs += tempval * 1.0 / samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
					model.add(range);
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
	cplex.end();
	model.end();
	cuts.end();
	center_cons.end();
	x.end();
	stat.finalpartitionsize = partition.size();
	if (option == 1)
		cout << "end solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "end evaluation mode..." << endl;
	return returnflag;
}
