import condition_evaluator as ce
from ..utils import gate_evaluator as ge
from ..utils import gate_simulator as gs
from ..utils import hardware_attenuation as ha
from ..utils import param_object as po
import trainer as tr
import result_saver as rs

samples = {
    "param": [],
    "param_attenuated": [],
    "score": [],
    "condition_closeness": []
}

loop_cond = True
loop_itterations = 0
while loop_cond:
    if loop_itterations == 0:
        param = po.ParamObject()
    else:
        raise NotImplementedError("Not implemented yet")
        #choose sample via training strategy
    
    samples["param"].append(param)
    param = ha.attenuate(param)
    samples["param_attenuated"].append(param)

    condition_closeness = ce.evaluate_condition(param)
    samples["condition_closeness"].append(condition_closeness)

    result = gs.simulate(param)
    score = ge.evaluate(result)
    samples["score"].append(score)

    if loop_itterations%100 == 0:
        rs.save_result(samples)

rs.save_result(samples)
