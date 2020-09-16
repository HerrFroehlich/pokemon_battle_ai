from src.training.competitions import Competition_1vs1, AgentConfig

conf = AgentConfig()
c = Competition_1vs1(1000, conf)
c.run()