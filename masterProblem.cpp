/*
  Master problem of optimization.
  Interacts with subproblems to improve approximations.
*/

#include "MasterProblem.h"

  /*
    Constructor
  */
  MasterProblem::MasterProblem(/* input variables at least need option */) {
    // *** only solve_level
    meanxvals(meanenv);
    meanobj = solve_mean_value_model(TSLP& prob, IloEnv meanenv, IloNumArray meanxvals, const vector<int>& samples);
    meanxvals.end();
    meanobj.end();
      // quadratic master
      levelmodel(lenv);
      lx(lenv, prob.firstvarlb, prob.firstvarub);
      ltheta(lenv, -IloInfinity, IloInfinity);
    // ***

    model(env);
    x(env, prob.firstvarlb, prob.firstvarub);
    theta(env, 0, IloInfinity);
  }

  /*
    Destructor
  */
  MasterProblem::~MasterProblem() {
    cplex.end();
    model.end();
    // *** quadratic (only in solve_level)
    levelcplex.end();
  	levelmodel.end();
  	lx.end();
    // ***
    x.end();
  }

  /*
    Solves the master problem by calling the solve function
    of the IloCplex variable.
  */
  IloCplex& getCplex() {
    return cplex;
  }

  /*
    Set initial constraints for optimization problem.
  */
  void define_lp_model() {
    for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
  	{
      model.add(find_constraint(i));
  	}
    add_objective();
  }

  /*
    Set initial constraint in quadratic master problem.
  */
  void define_qp_model() {
    IloExpr lsum(lenv);
    for (int j = 0; j < prob.nbFirstVars; ++j)
  		lsum += lx[j] * prob.objcoef[j];
  	lsum += ltheta;
  	IloRange rangeub(lenv, -IloInfinity, lsum, IloInfinity);
  	levelmodel.add(rangeub);
  	lsum.end();

    IloObjective lobj = IloMinimize(lenv);
    levelmodel.add(lobj);

    levelcplex(levelmodel);
    levelcplex.setParam(IloCplex::TiLim, 10800);
    levelcplex.setParam(IloCplex::Threads, 1);
    levelcplex.setParam(IloCplex::BarDisplay, 0);
    levelcplex.setOut(env.getNullStream());
  }

  /*
    Helper function for setting initial constraints.
  */
  IloRange find_constraint(int i) {
    IloExpr lhs(env);
    for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
      lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
    IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
    lhs.end();
    return range;
  }

  /*

  */
  void add_objective() {
    // Adding objective
  	IloExpr obj(env);
  	for (int i = 0; i < prob.nbFirstVars; ++i)
  		obj += x[i]*prob.objcoef[i];
  	// single cut
  	obj += theta;
  	model.add(IloMinimize(env, obj));
  	obj.end();
  	cplex(model);
  	cplex.setParam(IloCplex::TiLim,10800);
  	cplex.setParam(IloCplex::Threads, 1);
  	// Barrier
  	cplex.setParam(IloCplex::RootAlg, 2);
  	cplex.setParam(IloCplex::BarDisplay, 0);
  	cplex.setParam(IloCplex::SimDisplay, 0);
  	cplex.setOut(env.getNullStream());
  	//cplex.setParam(IloCplex::EpOpt, 1e-4);
  }

  void setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons) {
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

  double solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples) {
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
