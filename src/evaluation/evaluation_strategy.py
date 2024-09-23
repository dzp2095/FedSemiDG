class EvaluationStrategy:
    def validate(self, model, data_loader, device, save_path) -> dict:
        raise NotImplementedError
    
    def test(self, model, data_loader,  device, save_path) -> dict:
        raise NotImplementedError
    
    def custom_eval(self, model, data_loader, device, prefix, save_path) -> dict:
        raise NotImplementedError