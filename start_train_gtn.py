from library import Parameters
from library import LPSMLclass

para = Parameters.gtn()
GTN = LPSMLclass.GTN(para=para)
GTN.start_learning(learning_epochs=10)