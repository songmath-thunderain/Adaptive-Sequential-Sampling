#include "Partition.h"

//Took Out const vector<Component>& partition,
void Partition::setAggregatedBounds(const TSLP& prob, IloNumArray& secvarlb, IloNumArray& secvarub, IloNumArray& secconstrbd)
{
	for (int i = 0; i < partition.size(); ++i)
	{
		// set aggregated variable bound according to partition
		for (int j = 0; j < prob.nbSecVars; ++j)
		{
			bool lbinfflag = 0;
			bool ubinfflag = 0;
			double templb = 0.0;
			double tempub = 0.0;
			for (int k = 0; k < partition[i].indices.size(); ++k)
			{
				if (prob.secondvarlb[partition[i].indices[k] * prob.nbSecVars + j] == -IloInfinity)
				{
					lbinfflag = 1;
					break;
				}
				else
					templb += prob.secondvarlb[partition[i].indices[k] * prob.nbSecVars + j];
			}
			for (int k = 0; k < partition[i].indices.size(); ++k)
			{
				if (prob.secondvarub[partition[i].indices[k] * prob.nbSecVars + j] == IloInfinity)
				{
					ubinfflag = 1;
					break;
				}
				else
					tempub += prob.secondvarub[partition[i].indices[k] * prob.nbSecVars + j];
			}
			if (lbinfflag == 1)
				secvarlb[i * prob.nbSecVars + j] = -IloInfinity;
			else
				secvarlb[i * prob.nbSecVars + j] = templb;
			if (ubinfflag == 1)
				secvarub[i * prob.nbSecVars + j] = IloInfinity;
			else
				secvarub[i * prob.nbSecVars + j] = tempub;
		}
		// set second stage constraint bounds according to partition
		for (int d = 0; d < prob.nbSecRows; ++d)
		{
			double tempbd = 0.0;
			for (int k = 0; k < partition[i].indices.size(); ++k)
				tempbd += prob.secondconstrbd[partition[i].indices[k] * prob.nbSecRows + d];
			secconstrbd[i * prob.nbSecRows + d] = tempbd;
		}
	}
}

bool Partition::addToCollection(const VectorXf& dualvec, vector<DualInfo>& dualInfoCollection)
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

//Took Out const vector<Component>& partition,
void Partition::add_feas_cuts(IloEnv& env, TSLP& prob, IloModel& model, const IloNumVarArray& x, const IloNumArray& xvals, double subobjval, const VectorXf& dualvec, int i)
{
	// Add feasibility cuts
	vector<double> feas_cut_coef(prob.nbFirstVars, 0);
	double sum_xvals = 0.0;
	for (int ii = 0; ii < prob.nbSecRows; ++ii)
	{
		for (int j = 0; j < prob.nbPerRow[ii]; ++j)
		{
			if (prob.CoefInd[ii][j] < prob.nbFirstVars)
			{
				feas_cut_coef[prob.CoefInd[ii][j]] += prob.CoefMat[ii][j] * dualvec[ii] * partition[i].indices.size();
				sum_xvals += prob.CoefMat[ii][j] * dualvec[ii] * partition[i].indices.size() * xvals[prob.CoefInd[ii][j]];
			}
		}
	}
	IloExpr lhsfeas(env);
	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		if (fabs(feas_cut_coef[j]) > 1e-7)
			lhsfeas += feas_cut_coef[j] * x[j];
	}
	model.add(lhsfeas >= subobjval + sum_xvals);
	lhsfeas.end();
}

//Took Out const vector<Component>& partition,
double Partition::subprob_partition(IloNumArray& secvarlb, IloNumArray& secvarub, const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, int k, bool& feasflag)
{
	// Set constraint bounds
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		double bd = 0;
		for (int l = 0; l < partition[k].indices.size(); ++l)
		{
			double subbd = prob.secondconstrbd[partition[k].indices[l] * prob.nbSecRows + i];
			for (int j = 0; j < prob.nbPerRow[i]; ++j)
			{
				int ind = prob.CoefInd[i][j];
				if (ind < prob.nbFirstVars)
					subbd -= prob.CoefMat[i][j] * xvals[ind];
			}
			bd += subbd;
		}
		if (prob.secondconstrsense[i] == -1)
			subProb.suboptcon[i].setLB(bd);
		if (prob.secondconstrsense[i] == 0)
		{
			if (subProb.suboptcon[i].getUB() < bd)
			{
				subProb.suboptcon[i].setUB(bd);
				subProb.suboptcon[i].setLB(bd);
			}
			else
			{
				subProb.suboptcon[i].setLB(bd);
				subProb.suboptcon[i].setUB(bd);
			}
		}
		if (prob.secondconstrsense[i] == 1)
			subProb.suboptcon[i].setUB(bd);
	}
	// Set variable bounds
	for (int j = 0; j < prob.nbSecVars; ++j)
	{
		if (secvarlb[k * prob.nbSecVars + j] != -IloInfinity)
			subProb.suboptcon[prob.nbSecRows + j].setLB(secvarlb[k * prob.nbSecVars + j]);
		else
			subProb.suboptcon[prob.nbSecRows + j].setLB(-IloInfinity);
		if (secvarub[k * prob.nbSecVars + j] != IloInfinity)
			subProb.suboptcon[prob.nbSecRows + prob.nbSecVars + j].setLB(-secvarub[k * prob.nbSecVars + j]);
		else
			subProb.suboptcon[prob.nbSecRows + prob.nbSecVars + j].setLB(-IloInfinity);
	}
	subProb.suboptcplex.solve();
	double returnval;
	if (subProb.suboptcplex.getStatus() == IloAlgorithm::Optimal)
	{
		returnval = subProb.suboptcplex.getObjValue();
		subProb.suboptcplex.getDuals(duals, subProb.suboptcon);
		feasflag = 1;
	}
	else
	{
		feasflag = 0;
		// infeasible! Get extreme rays
		for (int i = 0; i < prob.nbSecRows; ++i)
		{
			double bd = 0;
			for (int l = 0; l < partition[k].indices.size(); ++l)
			{
				double subbd = prob.secondconstrbd[partition[k].indices[l] * prob.nbSecRows + i];
				for (int j = 0; j < prob.nbPerRow[i]; ++j)
				{
					int ind = prob.CoefInd[i][j];
					if (ind < prob.nbFirstVars)
						subbd -= prob.CoefMat[i][j] * xvals[ind];
				}
				bd += subbd;
			}
			if (prob.secondconstrsense[i] == -1)
				subProb.subfeascon[i].setLB(bd);
			if (prob.secondconstrsense[i] == 0)
			{
				if (subProb.subfeascon[i].getUB() < bd)
				{
					subProb.subfeascon[i].setUB(bd);
					subProb.subfeascon[i].setLB(bd);
				}
				else
				{
					subProb.subfeascon[i].setLB(bd);
					subProb.subfeascon[i].setUB(bd);
				}
			}
			if (prob.secondconstrsense[i] == 1)
				subProb.subfeascon[i].setUB(bd);
		}
		for (int j = 0; j < prob.nbSecVars; ++j)
		{
			if (secvarlb[k * prob.nbSecVars + j] != -IloInfinity)
				subProb.subfeascon[prob.nbSecRows + j].setLB(secvarlb[k * prob.nbSecVars + j]);
			else
				subProb.subfeascon[prob.nbSecRows + j].setLB(-IloInfinity);
			if (secvarub[k * prob.nbSecVars + j] != IloInfinity)
				subProb.subfeascon[prob.nbSecRows + prob.nbSecVars + j].setLB(-secvarub[k * prob.nbSecVars + j]);
			else
				subProb.subfeascon[prob.nbSecRows + prob.nbSecVars + j].setLB(-IloInfinity);
		}
		subProb.subfeascplex.solve();
		subProb.subfeascplex.getDuals(duals, subProb.subfeascon);
		returnval = subProb.subfeascplex.getObjValue();
	}
	return returnval;
}

void Partition::simple_refine(const Component& component, const TSLP& prob, const vector<IloNumArray>& extreme_points, const vector<int>& extreme_points_ind, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, vector<Component>& new_partition, vector< vector<int> >& extreme_ray_map)
{
	// Simple refinement strategy by putting vectors together according to a distance threshold
	vector<int> index_points_represent, index_rays_represent; // list of representing indices
	vector< vector<int> > index_points, index_rays; // index globally defined
	for (int i = 0; i < extreme_points.size(); ++i)
	{
		if (i == 0)
		{
			vector<int> newlist;
			newlist.push_back(extreme_points_ind[i]);
			index_points_represent.push_back(i);
			index_points.push_back(newlist);
		}
		else
		{
			bool distinctflag = 1;
			for (int j = 0; j < index_points_represent.size(); ++j)
			{
				bool tempflag = compare_arrays(prob, extreme_points[index_points_represent[j]], extreme_points[i]);
				if (tempflag == 1)
				{
					index_points[j].push_back(extreme_points_ind[i]);
					distinctflag = 0;
					break;
				}
			}
			if (distinctflag == 1)
			{
				vector<int> newlist;
				newlist.push_back(extreme_points_ind[i]);
				index_points_represent.push_back(i);
				index_points.push_back(newlist);
			}
		}
	}

	for (int i = 0; i < extreme_rays.size(); ++i)
	{
		if (i == 0)
		{
			// Since this is the first one, create a group starting with this one
			vector<int> newlist;
			newlist.push_back(extreme_rays_ind[i]);
			index_rays_represent.push_back(i);
			index_rays.push_back(newlist);
			vector<int> temp;
			temp.push_back(i);
			extreme_ray_map.push_back(temp);
		}
		else
		{
			bool distinctflag = 1;
			for (int j = 0; j < index_rays_represent.size(); ++j)
			{
				bool tempflag = compare_arrays(prob, extreme_rays[index_rays_represent[j]], extreme_rays[i]);
				if (tempflag == 1)
				{
					index_rays[j].push_back(extreme_rays_ind[i]);
					extreme_ray_map[j].push_back(i);
					distinctflag = 0;
					break;
				}
			}
			if (distinctflag == 1)
			{
				vector<int> newlist;
				newlist.push_back(extreme_rays_ind[i]);
				index_rays_represent.push_back(i);
				index_rays.push_back(newlist);
				vector<int> temp;
				temp.push_back(i);
				extreme_ray_map.push_back(temp);
			}
		}
	}
	for (int j = 0; j < index_rays.size(); ++j)
	{
		Component compo;
		compo.indices = index_rays[j];
		new_partition.push_back(compo);
	}
	for (int j = 0; j < index_points.size(); ++j)
	{
		Component compo;
		compo.indices = index_points[j];
		new_partition.push_back(compo);
	}
}

bool Partition::compare_arrays(const TSLP& prob, const IloNumArray& array1, const IloNumArray& array2)
{
	// check if they are "equal" to each other
	bool returnflag = 1;
	for (int j = 0; j < array1.getSize(); ++j)
	{
		if (fabs((array1[j] - array2[j]) * 1.0 / (array1[j] + 1e-5)) > prob.distinct_par)
		{
			returnflag = 0;
			break;
		}
	}
	return returnflag;
}

void Partition::gen_feasibility_cuts(IloEnv& env, const TSLP& prob, const IloNumArray& xvals, const vector<int>& extreme_ray_map, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, const double sum_of_infeas, IloModel& model, const IloNumVarArray& x)
{
	vector<double> feas_cut_coef(prob.nbFirstVars, 0);
	double sum_xvals = 0.0;
	for (int j = 0; j < extreme_ray_map.size(); ++j)
	{
		int ind = extreme_ray_map[j];
		for (int i = 0; i < prob.nbSecRows; ++i)
		{
			for (int j = 0; j < prob.nbPerRow[i]; ++j)
			{
				if (prob.CoefInd[i][j] < prob.nbFirstVars)
				{
					feas_cut_coef[prob.CoefInd[i][j]] += prob.CoefMat[i][j] * extreme_rays[ind][i];
					sum_xvals += prob.CoefMat[i][j] * extreme_rays[ind][i] * xvals[prob.CoefInd[i][j]];
				}
			}
		}
	}
	IloExpr lhs(env);
	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		if (fabs(feas_cut_coef[j]) > 1e-7)
			lhs += x[j] * feas_cut_coef[j];
	}
	model.add(lhs >= sum_of_infeas + sum_xvals);
	lhs.end();
}