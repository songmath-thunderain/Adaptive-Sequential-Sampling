/*
  Master problem of optimization.
  Can be updated to a better approximation using subproblems.

*/

#include "masterProblem.h"

  // Constructor
  masterProblem::masterProblem(/* input variables at least need option */) {

  }

  void masterProblem::subProblem() {
    // Level: level method starts with the mean-value solution
  	// option = 0: solving SAA for getting a candidate solution; option = 1: solving SAA for evaluating a given solution using CI
  	// first solve a mean-value problem
    // option = -1: solving single cut, leave out *********explanation***********
  	double accuracy;
  	if (option == 0 || option == -1) // add single cut (option 0 from main)
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
    // ONLY LEVEL
    if (option == 0 || option == 1) {
      IloEnv meanenv;
      IloNumArray meanxvals(meanenv);
      double meanobj = solve_mean_value_model(prob, meanenv, meanxvals, samples);
      vector<double> xiterate(prob.nbFirstVars);
      // Assign xiterate to meanx
      for (int j = 0; j < prob.nbFirstVars; ++j)
        xiterate[j] = meanxvals[j];
      // *********Order matter for these??*********
      meanxvals.end();
      meanenv.end();
    }
    bool feas_flag = 1;

    // Define Master program
  	IloModel mastermodel(env);
  	IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
  	// For now we assume an LB for the second-stage obj: 0
  	IloNumVar theta(env, 0, IloInfinity);
  	// Adding first-stage constraints
  	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
  	{
  		IloExpr lhs(env);
  		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
  			lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
  		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
  		mastermodel.add(range);
  		lhs.end();
  	}

    // Adding objective
  	IloExpr obj(env);
  	for (int i = 0; i < prob.nbFirstVars; ++i)
  		obj += x[i]*prob.objcoef[i];
  	// single cut
  	obj += theta;
  	mastermodel.add(IloMinimize(env, obj));
  	obj.end();
  	IloCplex mastercplex(mastermodel);
  	mastercplex.setParam(IloCplex::TiLim,10800);
  	mastercplex.setParam(IloCplex::Threads, 1);
  	// Barrier
  	mastercplex.setParam(IloCplex::BarDisplay, 0);
  	mastercplex.setParam(IloCplex::SimDisplay, 0);
    // This parameter is only in option 0 ??? *********
    if (option = -1)
  	 mastercplex.setParam(IloCplex::RootAlg, 2);

  	mastercplex.setOut(env.getNullStream());
  	//mastercplex.setParam(IloCplex::EpOpt, 1e-4);

    // Can separate into helper funtion ???
    if (option == 0 || option == 1) {
      // Define Level quadratic program
  	 IloEnv lenv;
  	 IloModel levelmodel(lenv);
  	 IloNumVarArray lx(lenv, prob.firstvarlb, prob.firstvarub);
  	 IloNumVar ltheta(lenv, -IloInfinity, IloInfinity);
  	 // Adding first-stage constraints
  	 for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
  	 {
  		  IloExpr lhs(lenv);
  		  for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
  			 lhs += lx[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
  		  IloRange range(lenv, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
  		  levelmodel.add(range);
  		  lhs.end();
  	 }
  	 // define the first constraint for level master
  	 IloExpr lsum(lenv);
  	 for (int j = 0; j < prob.nbFirstVars; ++j)
  		  lsum += lx[j]*prob.objcoef[j];
  	   lsum += ltheta;
  	 IloRange rangeub(lenv, -IloInfinity, lsum, IloInfinity);
  	 levelmodel.add(rangeub);
  	 lsum.end();

  	 IloObjective lobj = IloMinimize(lenv);
  	 levelmodel.add(lobj);

  	 IloCplex levelcplex(levelmodel);
  	 levelcplex.setParam(IloCplex::TiLim,10800);
  	 levelcplex.setParam(IloCplex::Threads, 1);
  	 levelcplex.setParam(IloCplex::BarDisplay, 0);
  	 levelcplex.setOut(env.getNullStream());
    }


    // Loop to get desired accuracy
    while ((stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) > accuracy || feas_flag == 0) {
      if (option == 0 || option == 1) {
        singleCut();
      }
      else if (option == -1) {
        level();
      }
    }

    // Clear memory
    subp.suboptcplex.end();
    subp.suboptmodel.end();
    subp.suboptcon.end();
    subp.subopty.end();
    subp.subfeascplex.end();
    subp.subfeasmodel.end();
    subp.subfeascon.end();
    subp.subfeasy.end();
    mastercplex.end();
    mastermodel.end();
    if (option == 0 || option == -1) {
    	levelcplex.end();
    	levelmodel.end();
    	lx.end();
    }
    x.end();
  }

  void masterProblem::singleCut() {
      feas_flag = 1;
      stat.iter++;
      IloNumArray xvals(env);
      double thetaval;
      double lasttime = clock.getTime();
      mastercplex.solve();
      cout << "master status = " << mastercplex.getStatus() << endl;
      if (mastercplex.getStatus() == IloAlgorithm::Unbounded)
      {
        IloEnv meanenv;
        stat.relaxobjval = solve_mean_value_model(prob, meanenv, xvals, samples);
        meanenv.end();
      }
      else
      {
        stat.relaxobjval = mastercplex.getObjValue();
        mastercplex.getValues(xvals, x);
        thetaval = mastercplex.getValue(theta);
      }
      for (int j = 0; j < prob.nbFirstVars; ++j)
        xiterateXf(j) = xvals[j];
      stat.mastertime += clock.getTime()-lasttime;
      double feasbound = 0.0;
      IloExpr lhsaggr(env);
      double lhsaggrval = 0;
      lhsaggr += theta;
      lhsaggrval += thetaval;
      double rhsaggr = 0;
      lasttime = clock.getTime();
      for (int k = 0; k < nbScens; ++k)
      {
        // solve subproblems for each scenario
        IloNumArray duals(env);
        bool feasflag;
        double subobjval = subprob(subp, prob, xvals, duals, samples[k], feasflag);
        VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
        for (int i = 0; i < prob.nbSecRows+prob.nbSecVars; ++i)
          dualvec(i) = duals[i];
        duals.end();
        if (feasflag == 1)
        {
          // optimal, so return extreme point solution
          VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
          double sum_xvals = 0;
          for (int j = 0; j < prob.nbFirstVars; ++j)
          {
            if (fabs(opt_cut_coef[j]) > 1e-7)
            {
              lhsaggrval += xvals[j]*opt_cut_coef[j]*1.0/nbScens;
              lhsaggr += x[j]*opt_cut_coef[j]*1.0/nbScens;
              sum_xvals += opt_cut_coef[j]*xvals[j];
            }
          }
          double rhssub = subobjval + sum_xvals;
          rhsaggr += rhssub*1.0/nbScens;
          feasbound += subobjval;
        }
        else
        {
          cout << "infeasible!" << endl;
          feas_flag = 0;
          // infeasible, so return extreme rays
          VectorXf feas_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0, prob.nbSecRows);
          double sum_xvals = feas_cut_coef.dot(xiterateXf);
          IloExpr lhssub(env);
          for (int j = 0; j < prob.nbFirstVars; ++j)
          {
            if (fabs(feas_cut_coef[j]) > 1e-7)
              lhssub += x[j]*feas_cut_coef[j];
          }
          double rhssub = sum_xvals+subobjval;
          stat.num_feas_cuts++;
          mastermodel.add(lhssub >= rhssub);
          lhssub.end();
        }
      }
      stat.subtime += clock.getTime()-lasttime;
      if (feas_flag == 1)
      {
        feasbound = feasbound*1.0/nbScens;
        for (int j = 0; j < prob.nbFirstVars; ++j)
          feasbound += xvals[j]*prob.objcoef[j];
        if (feasbound <= stat.feasobjval)
          stat.feasobjval = feasbound;
        mastermodel.add(lhsaggr >= rhsaggr);
        stat.num_opt_cuts++;
      }
      else
        cout << "Infeasible! Generating feasibility cuts!" << endl;
      lhsaggr.end();
      xvals.end();
      cout << "relaxobjval = " << stat.relaxobjval << endl;
      cout << "feasobjval = " << stat.feasobjval << endl;
      cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
      stat.solvetime = clock.getTime()-starttime;
      cout << "stat.solvetime = " << stat.solvetime << endl;
      if (stat.solvetime > 10800)
        break;
  }

  void masterProblem::level() {
    feas_flag = 1;
    IloEnv env2;
    stat.iter++;
    IloNumArray xiteratevals(env2, prob.nbFirstVars);
    for (int j = 0; j < prob.nbFirstVars; ++j)
    {
      xiterateXf(j) = xiterate[j];
      xiteratevals[j] = xiterate[j];
    }
    double feasbound = 0.0;
    for (int j = 0; j < prob.nbFirstVars; ++j)
    feasbound += xiterate[j]*prob.objcoef[j];
    IloExpr lhsaggr(env);
    lhsaggr += theta;
    IloExpr llhsaggr(lenv);
    llhsaggr += ltheta;
    double rhsaggr = 0;
    double lasttime = clock.getTime();
    for (int k = 0; k < nbScens; ++k)
    {
      // solve subproblems for each partition
      IloNumArray duals(env2);
      bool feasflag;
      double subobjval = subprob(subp, prob, xiteratevals, duals, samples[k], feasflag);
      VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
      for (int i = 0; i < prob.nbSecRows+prob.nbSecVars; ++i)
        dualvec(i) = duals[i];
      duals.end();
      if (feasflag == 1)
      {
        // optimal, so return extreme point solution
        VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
        double sum_xvals = opt_cut_coef.dot(xiterateXf);
        double rhssub = subobjval + sum_xvals;
        feasbound += subobjval*1.0/nbScens;
        rhsaggr += rhssub*1.0/nbScens;
        for (int j = 0; j < prob.nbFirstVars; ++j)
        {
          if (fabs(opt_cut_coef[j]) > 1e-7)
          {
            lhsaggr += x[j]*opt_cut_coef[j]*1.0/nbScens;
            llhsaggr += lx[j]*opt_cut_coef[j]*1.0/nbScens;
          }
        }

      }
      else
      {
        feas_flag = 0;
        // infeasible, so return extreme rays
        VectorXf feas_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0, prob.nbSecRows);
        double sum_xvals = feas_cut_coef.dot(xiterateXf);
        IloExpr lhssub(env);
        IloExpr llhssub(lenv);
        for (int j = 0; j < prob.nbFirstVars; ++j)
        {
          if (fabs(feas_cut_coef[j]) > 1e-7)
          {
            llhssub += lx[j]*feas_cut_coef[j];
            lhssub += x[j]*feas_cut_coef[j];
          }
        }
        double rhssub = sum_xvals+subobjval;
        stat.num_feas_cuts++;
        mastermodel.add(lhssub >= rhssub);
        levelmodel.add(llhssub >= rhssub);
        lhssub.end();
        llhssub.end();
      }
    }
    xiteratevals.end();
    stat.subtime += clock.getTime()-lasttime;
    if (feas_flag == 1)
    {
    //cout << "feasbound = " << feasbound << ", feasobjval = " << stat.feasobjval << endl;
      if (feasbound <= stat.feasobjval)
        stat.feasobjval = feasbound;
      levelmodel.add(llhsaggr >= rhsaggr);
      mastermodel.add(lhsaggr >= rhsaggr);
      stat.num_opt_cuts++;
    }
    lhsaggr.end();
    llhsaggr.end();
    // Now solve the master, get a lower bound
    lasttime = clock.getTime();
    mastercplex.solve();
    stat.relaxobjval = mastercplex.getObjValue();
    stat.mastertime += clock.getTime()-lasttime;

    // Now solve the qp level problem
    // update the upper bound, (1-\lambda)F^{k+1}+\lambda F^*
    rangeub.setUB(0.5*stat.relaxobjval + 0.5*stat.feasobjval);
    IloExpr objExpr(lenv);
    for (int j = 0; j < prob.nbFirstVars; ++j)
    {
      objExpr += lx[j]*lx[j];
      objExpr -= lx[j]*2*xiterate[j];
    }
    lobj.setExpr(objExpr);
    objExpr.end();

    double startqptime = clock.getTime();
    levelcplex.solve();
    stat.qptime += clock.getTime()-startqptime;
    IloNumArray lxval(lenv);
    levelcplex.getValues(lxval, lx);
    for (int j = 0; j < prob.nbFirstVars; ++j)
      xiterate[j] = lxval[j];
    lxval.end();
    env2.end();
    //cout << "relaxobjval = " << stat.relaxobjval << endl;
    //cout << "feasobjval = " << stat.feasobjval << endl;
    //cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
    stat.solvetime = clock.getTime()-starttime;
    if (stat.solvetime > 10800)
      break;
  }
