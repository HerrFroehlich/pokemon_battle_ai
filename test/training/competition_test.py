from src.training.competitions import Competition_1vs1, CompetitionConfig

conf = CompetitionConfig(N_BATTLES=1000, N_BATTLES_WITH_SAME_TEAM=10)

conf.AGENTCONFIG.BATCH_SIZE = 32

print(conf)
c = Competition_1vs1(1000, conf)
c.run()