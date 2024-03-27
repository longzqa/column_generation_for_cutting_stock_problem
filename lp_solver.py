# -*- coding:utf-8 -*-
""" ortools 求解线性规划, 整数规划模型"""

import os
import time
import numpy
import numpy as np
from ortools.linear_solver import pywraplp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


class LPProblem:
    _default_type = 'lp'
    _enum_type = {'lp', 'ip'}

    def __init__(self, obj_cof, obj_cons, mat_cof, lbs, ubs, prob_name='', prob_type='lp'):
        """

        :param prob_name: str,
        :param obj_cof: 目标系数, [..., -1/1], -1-min, 1-max
        :param obj_cons: 目标函数中的常数项
        :param mat_cof: 约束矩阵(不包含变量的取值范围),
                        [..., -1/0/1, b], -1-小于等于, 1-大于等于, 0-等于
        :param lbs: 变量最小值(包含该值)
        :param ubs: 变量最大值
        :param prob_type: 问题类型, lp/ip

        example obj_cof
        ---------------
        [1, 3, 0, -1] 表示 min(x1 + 3*x2 + 0*x3)

        example mat_cof
        ---------------
        [
        [2, 0, 0, -1, 9],
        [0, 1, 1, 0, 20]
        ] 表示 2*x1 < 9, x2 + x3 = 20

        example lbs
        ------------
        [1, 2, 3, 4] 表示x1最小值1, x2最小值2, x3最小值3, x4最小值4

        """
        self.prob_name = prob_name
        self.prob_type = prob_type if prob_type in LPProblem._enum_type else LPProblem._default_type
        self.obj_cof = LPProblem._np_to_list(obj_cof)
        self.obj_cons = obj_cons
        self.mat_cof = LPProblem._np_to_list(mat_cof)
        self.lbs = LPProblem._np_to_list(lbs)
        self.ubs = LPProblem._np_to_list(ubs)

    @staticmethod
    def _np_to_list(data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data


class LPSolver:
    def __init__(self, prob: LPProblem):
        self.prob_name = prob.prob_name
        self.prob_type = prob.prob_type
        self.obj_cof = prob.obj_cof
        self.obj_cons = prob.obj_cons
        self.mat_cof = prob.mat_cof
        self.lbs = prob.lbs
        self.ubs = prob.ubs

        '''
        empty: 未开始求解
        invalid: 模型无效
        fail: 求解失败

        feasible: 只找到可行解
        infeasible: 未找到可行解
        optimal: 找到最优解
        '''
        self.status = "empty"
        self.obj = 0
        self.solution = []
        self.dual_val = []  # 对偶变量/影子价格

    def solve(self, solver_id=None, save_mod=False):
        """

        :param solver_id: str, 指定的求解器
        :param save_mod: bool, 是否需要将模型保存下来, data/20240308_1350_{prob_name}.lp
        :return:
        """

        def _add_constraint(solver, expr, _sign, _b, cons_lst):
            expr = solver.Sum(expr)
            if _sign == 0:
                c = solver.Add(expr == _b)
            elif _sign < 0:
                c = solver.Add(expr <= _b)
            else:
                c = solver.Add(expr >= _b)
            cons_lst.append(c)

        # 检查模型是否有效
        if not self._is_valid():
            self.status = 'invalid'
            return
        num_var = len(self.obj_cof) - 1

        # 声明求解器
        if solver_id:
            sol = pywraplp.Solver.CreateSolver(solver_id)
        elif self.prob_type == 'lp':
            sol = pywraplp.Solver.CreateSolver('GLOP')
        else:
            sol = pywraplp.Solver.CreateSolver('SCIP')

        # 创建连续/整型变量
        if self.prob_type == 'ip':
            x = [sol.IntVar(lb=self.lbs[i], ub=self.ubs[i], name=f'x_{i}') for i in range(num_var)]
        else:
            x = [sol.NumVar(lb=self.lbs[i], ub=self.ubs[i], name=f'x_{i}') for i in range(num_var)]

        # 目标函数
        obj_expr = [self.obj_cof[i] * x[i] for i in range(num_var)]
        obj_expr += [self.obj_cons]
        sol.Maximize(sol.Sum(obj_expr)) if self.obj_cof[-1] >= 0 else sol.Minimize(sol.Sum(obj_expr))

        # 添加约束矩阵约束
        cons_list = []
        for cons in self.mat_cof:
            cons_expr = [cons[i] * x[i] for i in range(num_var)]
            sign, b = cons[-2:]
            _add_constraint(sol, cons_expr, sign, b, cons_list)

        # 保存lp文件
        if save_mod:
            prefix = time.strftime('%Y%m%d_%H%M_')
            lp_file = os.path.join(DATA_DIR, prefix + self.prob_name + '.lp')
            with open(lp_file, 'w', encoding='utf-8') as fw:
                fw.write(sol.ExportModelAsLpFormat(True))
                fw.write('/n')
                fw.write(self.prob_name)

        # 求解
        status = sol.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            self.status = 'optimal'
            self.obj = sol.Objective().Value()
            self.solution = [x[i].solution_value() for i in range(num_var)]
            if self.prob_type == 'lp':  # 只有lp才有对偶变量
                self.dual_val = [c.dual_value() for c in cons_list]
        elif status == pywraplp.Solver.FEASIBLE:
            self.status = 'feasible'
        elif status == pywraplp.Solver.INFEASIBLE:
            self.status = 'infeasible'
        else:
            self.status = 'fail'

        print(f'++++++++++++++prob_name:{self.prob_name}')
        print(f'++++++++++++++++++++obj:{self.status}')
        print(f'++++++++++++++++++++obj:{self.obj}')
        print(f'++++++++++++++++++++sol:{self.solution}')
        print(f'+++++++++++++++dual_val:{self.dual_val}\n')
        # sol.parameters.max_time_in_seconds()

    def _is_valid(self):
        """ 检查模型是否有效"""
        # 检查目标系数是否包含变量系数
        if len(self.obj_cof) <= 1:
            print(f'模型目标系数未包含变量系数, obj_cof:{self.obj_cof}')
            return False

        # 检查变量最小最大值数量、是否出现最小最大值不可行的情况
        var_num = len(self.obj_cof) - 1
        if len(self.lbs) != var_num or len(self.ubs) != var_num:
            print(f'变量取值范围参数有误, var_num:{var_num}, lbs:{self.lbs}, ubs:{self.ubs}')
            return False
        for i in range(var_num):
            if self.lbs[i] > self.ubs[i]:
                print(f'变量最小值大于最大值, i:{i}, lbs:{self.lbs}, ubs:{self.ubs}')
                return

        # 检查约束矩阵的系数个数
        for i in range(len(self.mat_cof)):
            cons = self.mat_cof[i]
            if len(cons) != var_num + 2:
                print(f'模型约束系数个数有误, mat_cof:{self.mat_cof}, i:{i}')
                return False
        return True


if __name__ == '__main__':
    _obj = [5, 5, 5, -1]
    mat = [
        [2, 0, 0, 1, 30],
        [0, 1, 0, 1, 20],
        [0, 0, 1, 1, 40],
    ]
    _lbs = [0, 0, 0]
    _ubs = [50, 50, 50]
    name = 'first'
    p = LPProblem(_obj, 0, mat, _lbs, _ubs, name, 'ip')
    s = LPSolver(p)
    s.solve(save_mod=True)
