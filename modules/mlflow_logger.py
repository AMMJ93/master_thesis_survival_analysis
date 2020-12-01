import mlflow


class Logger:
    """
    Logger for MLFlow
    """
    def __init__(self, url, experiment):
        self.url = url
        self.experiment = experiment
        mlflow.set_tracking_uri(self.url)
        mlflow.set_experiment(self.experiment)

    def log_metrics(self, log, run_name, params):
        assert isinstance(params, object), "Params must be dictionary"
        log_df = log.to_pandas()
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            for epoch in range(len(log_df)):
                train = log_df.loc[epoch, 'train_loss']
                val = log_df.loc[epoch, 'val_loss']
                mlflow.log_metric(key="train_loss", value=train, step=epoch)
                mlflow.log_metric(key="val_loss", value=val, step=epoch)
            mlflow.end_run()

    def log_predictions(self, df, run_name, concordance, brier, params, targets, groups):
        assert isinstance(params, object), "Params must be dictionary"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            for i in range(groups):
                idx = targets == i
                group_df = df.loc[:, idx].mean(axis=1).rename(i)
                for idx in group_df.index.tolist():
                    pred = group_df.loc[idx]
                    mlflow.log_metric(key='group-{}'.format(i), value=pred, step=int(idx))
            mlflow.log_metric(key="integr_brier", value=brier)
            mlflow.log_metric(key="concordance", value=concordance)
            mlflow.end_run()