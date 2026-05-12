- # titolo (#, ##, ###…)
 - **testo** grassetto
 - *testo* corsivo
 - ~~testo~~ barrato
 - `codice` codice inline
 - ```linguaggio ... ``` blocco di codice
 - - oppure * elenco puntato
 - 1. elenco numerato
 - [testo](url) link
 - ![alt](url) immagine
 - > citazione
 - --- linea orizzontale
 - | tabelle (con --- nella riga intestazione)

 ```python
 ```

- [Flowe Framework](#flowe-framework)
    - [Install](#install)
  - [Config Record](#config-record)
      - [Dynamic Configuration of ConfigRecord](#dynamic-configuration-of-configrecord)
  - [Design Stateful Client](#design-stateful-client)
    - [Saving Model Parameters to the context](#saving-model-parameters-to-the-context)
      - [Saving NumPy arrays to the context](#saving-numpy-arrays-to-the-context)
      - [Saving PyTorch parameters to the context](#saving-pytorch-parameters-to-the-context)
  - [Use Strategies](#use-strategies)
    - [Using Start Method](#using-start-method)
  - [Aggregate Evaluation Results](#aggregate-evaluation-results)
    - [Using a custom metrics aggregation function](#using-a-custom-metrics-aggregation-function)
  - [How to save model checkpoints in ServerApp](#how-to-save-model-checkpoints-in-serverapp)


# Flowe Framework (How-to-Guides)
### Install
python -m pip install flwr\
python -m pip install "flwr[simulation]"\
pip install -e .\
flwr run . –stream

## Config Record
By using a ConfigRecord, values from serverapp can be sent to clientapp via the Message. ConfigRecord can be sent as part of the Message by passing it to the start of the choose strategy. By default it contains a key server-round so the client know the current round of the FL process. Value passed via start method are static, except the server-round.

 ```python
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    strategy = FedAvg()

    config = ConfigRecord({"lr": 0.1, "optim": "adam-w", "augment": True})

        result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=config,
        num_rounds=10,
    )
 ```

 #### Dynamic Configuration of ConfigRecord
 Can be introduced by implementing a custom strategy that overrides the `configure_train` method. The `configure_train` is responsible for among other aspects, to create the Messages that will be sent to the ClientApps. Message typically contains:
- array record with parameters of the model
- config record containing configurations used by the clientapp

 ```python
class CustomFedAdagrad(FedAvg):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        # Note: server_round starts at 1, not 0
        if server_round % 5 == 0:
            config["lr"] *= 0.5
            print("LR decreased to:", config["lr"])
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)
```

## Design Stateful Client
By default clients are stateless, client objects are created again at each round and a new message is sent. But the **context is unique to each client** meaning that subsequent executions of the same ClientApp from the same node will receive the same Context object. The context contains (not only this).
- **context.run_config**: defaults come from [tool.flwr.app.config] in pyproject.toml, then can be overridden via flwr run . --run-config … (contains partition-id, num-partitions, and more???).
- **context.node_config**: provided by the runtime/simulation (e.g., partition-id, num-partitions), not from your pyproject.toml.
- **context.state**: attribute (type of RecordDict) can be used to store information that you would like the ClientApp to have access to for the duration of the run.

 **TIP:** Recall, the state attribute of a Context object is of type RecordDict, which is a special dictionary for different types of records available in Flower. This means that you can save to it not just MetricRecord as in the example below, but also ArrayRecord and ConfigRecord object

  ```python
# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    n = random.randint(0, 10)
    print(n)

    # Append to list in context or initialize if it doesn't exist
    if "random-metrics" not in context.state:
        # Initialize MetricRecord in state
        context.state["random-metrics"] = MetricRecord({"random-ints": []})

    # Append to record
    context.state["random-metrics"]["random-ints"].append(n)

    # Print history
    print(context.state["random-metrics"])
    return Message(RecordDict(), reply_to=msg)
```

### Saving Model Parameters to the context
Flower has some custom records that can be used like:
- ConfigRecords/Metric records to store float, integers, boolean, string, bytes and simple componens in general
- ArraysRecord to store model parameters or data arrays. Numpy arrays need to be serialized and then deserialized

#### Saving NumPy arrays to the context
```python
import numpy as np
from flwr.app import Array, ArrayRecord, Context


# Let's create a simple NumPy array
arr_np = np.random.randn(3, 3)

# Now, let's serialize it and construct an Array
arr = Array(arr_np)

# It can be inserted in an ArrayRecord like this
arr_record = ArrayRecord()
arr_record["my_array"] = arr
# You can also do it via the constructor
# arr_record = ArrayRecord({"my_array": arr})

# If you don't need the keys, you can also pass a list of Numpy arrays
# arr_record = ArrayRecord([arr_np])

# Then, it can be added to the state in the context
context.state["some_parameters"] = arr_record
```
To extract the data in an ArrayRecord, you just need to deserialize the array of interest. For example, following the example above:

```python
# Get Array from context
arr = context.state["some_parameters"]["my_array"]

# If you constructed the ArrayRecord with a list of Numpy, then do
# arr = context.state["some_parameters"].to_numpy_ndarrays()[0]  # get first array

# Deserialize it
arr_deserialized = arr.numpy()
```
#### Saving PyTorch parameters to the context


```python
model = Net()

# Save the state_dict into a single ArrayRecord
arr_record = ArrayRecord(model.state_dict())

# Add to a context
context.state["net_parameters"] = arr_record
```


```python
state_dict = {}
arr_record = context.state["net_parameters"]

# Deserialize the parameters
state_dict = arr_record.to_torch_state_dict()

# Apply state dict to a new model instance
model_ = Net()
model_.load_state_dict(state_dict)
```
## Use Strategies
it is possible to:
- Use an existing strategy, for example, FedAvg
- Customize an existing strategy with callback functions to its start method
- Customize an existing strategy by overriding one or more of its methods.
- Implement a novel strategy from scratch

To use an existing strategy the start method is used, every start method of each strategy has some common parameters and some specific ones.

```python
strategy = FedAvg(
    fraction_train=0.5,  # fraction of nodes to involve in a round of training
    fraction_evaluate=1.0,  # fraction of nodes to involve in a round of evaluation
    min_available_nodes=100,  # minimum connected nodes required before FL starts
)
```
It is also possible to customize the name of the parameters instead of using the default ones.

```python
strategy = FedAvg(
    arrayrecord_key="my-arrays",
    configrecord_key="super-config",
    weighted_by_key="num-batches",
)
```

- `arrayrecord_key`: the Message communicated to the ClientApp will contain an ArrayRecord containing the arrays of the global model under this key. By default the key is "arrays".
- `configrecord_key`: the Message communicated to the ClientApp will contain a ConfigRecord containing config settings. By default the key is "config".
- `weighted_by_key`: A key inside the MetricRecord that the ClientApp returns as part of its reply to the ServerApp. The value under this key is used to perform weighted aggregation of MetricRecords and, after a round of federated training, ArrayRecords. The default value is "num-examples".

With a strategy defined as in the code snippet above, the ClientApp should receive a Message with the following structure:

```python
# The content of a Message arriving to the ClientApp will have
# the following structure and using the keys defined in the strategy
msg = Message(
    # ....
    content=RecordDict(
        {
            "my-arrays": ArrayRecord(...),
            "super-config": ConfigRecord(...),
        
}
    )
)

# The reply Message should contain a MetricRecord and inside it
# an item associated with the key used to initialize the strategy
reply_msg_content = RecordDict(
    {
        "locally-updated-params": ArrayRecord(...),
        "local-metrics": MetricRecord(
            {
                "num-batches": N,
                # ... Other metrics
            }
        ),
    }
)
```
### Using Start Method
The start method is the one that launch the federated learning process. It requires.
- Grid: is an object that will be used to interface with the nodes running the ClientApp to involve them in a round of train/evaluate/query or other. 
- ArrayRecord: contains the parameters of the model we want to federate

```python
# Define configs to send to ClientApp
train_cfg = ConfigRecord({"lr": 0.1, "optim": "adam"})
eval_cfg = ConfigRecord({"max-steps": 500, "local-checkpoint": True})

# Start strategy
result = strategy.start(
    grid=grid,
    initial_arrays=ArrayRecord(...),
    train_config=train_cfg,
    evaluate_config=eval_cfg,
    num_rounds=100,
)
```

The start method can trace an argument called `evaluate_fn`, and it allows passing to it a callback function to evaluate the aggregated model on some local data that the ServerApp might have access to. 

```python
# Callback definition. The function can have any name
# but the arguments are fixed
def my_callback(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Save checkpoint
    state_dict = arrays.to_torch_state_dict()
    torch.save(state_dict, f"model_at_round_{server_round}.pt")

    # eval model on local data
    model = MyModel()
    model.load_state_dict(state_dict)
    acc, loss = test(model, ...)

    # Return MetricRecord
    return MetricRecord({"acc": acc, "loss": loss})


# Pass the callback to the start method
strategy.start(..., evaluate_fn=my_callback)
```

## Aggregate Evaluation Results
Flower strategies automatically aggregate the metrics in the MetricRecord in the Messages replied by the ClientApps. By default, a weighted aggregation is performed for all metrics using as weight the value assigned to the `weighted_by_key` attribute of a strategy. By default `weighted_by_key="num-examples"`. 

Let’s see how we can define a custom aggregation function for MetricRecord objects received in the reply of an evaluation round.

```python
strategy = FedAvg(
    # ... other parameters ...
    weighted_by_key="your-key",  # Key to use for weighted averaging
    evaluate_metrics_aggr_fn=my_metrics_aggr_function,  # Custom aggregation function
)
```

**TIP**: In Flower, `evaluate_metrics_aggregation_fn` and `evaluate_fn` serve different purposes.
- `evaluate_metrics_aggregation_fn` (defined in the strategy) combines the evaluation metrics returned by clients at each round (for example, weighted average accuracy).
- `evaluate_fn` (passed to strategy.start(...)) performs a centralized server-side evaluation of the global model using a server dataset.

So, the first aggregates client-reported metrics, while the second directly evaluates the current global model.

### Using a custom metrics aggregation function
The `evaluate_metrics_aggr_fn` can be customized to support any evaluation results aggregation logic you need. Its definition is:

```python
Callable[[list[RecordDict], str], MetricRecord]
```
It takes a list of RecordDict and a weighting key as inputs and returns a MetricRecord. For example, the function below extracts and returns the minimum value for each metric key across all Message:

```python
from flwr.app import MetricRecord, RecordDict


def custom_metrics_aggregation_fn(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Extract the minimum value for each metric key."""
    aggregated_metrics = MetricRecord()

    # Track current minimum per key in a plain dict,
    # then copy into MetricRecord at the end
    mins = {}

    for record in records:
        for record_item in record.metric_records.values():
            for key, value in record_item.items():
                if key == weighting_metric_name:
                    # We exclude the weighting key from the aggregated MetricRecord
                    continue

                if key in mins:
                    if value < mins[key]:
                        mins[key] = value
                else:
                    mins[key] = value

    for key, value in mins.items():
        aggregated_metrics[key] = value

    return aggregated_metrics
```

## How to save model checkpoints in ServerApp
To save model checkpoints in ServerApp across different FL rounds, you can implement this in a customized evaluate_fn and pass it to the strategy’s start method. Here’s an example showing how to save the global PyTorch model:

```python
def get_evaluate_fn(save_every_round, total_round, save_path):
    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Save model every `save_every_round` round and for the last round
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Convert ArrayRecord to PyTorch state dict
            state_dict = arrays.to_torch_state_dict()

            # Save model weights to disk
            torch.save(state_dict, f"{save_path}/model_{server_round}.pt")

        return MetricRecord()

    return evaluate
```
Then, pass it to the start method of the defined strategy.


## Mods
A Mod is a callable that wraps around a ClientApp. It can manipulate or inspect the incoming Message and the resulting outgoing Message. The signature for a Mod is as follows:

```python
ClientAppCallable = Callable[[Message, Context], Message]
Mod = Callable[[Message, Context, ClientAppCallable], Message]
```
There are application-wide mods, these mods apply to all functions within the ClientApp, and function-specific-mods, these mods apply only to a specific function (e.g, the function decorated by @app.train()).

https://flower.ai/docs/framework/how-to-use-built-in-mods.html

# Flowe Framework (Explanations)
## Federated Evaluation
https://flower.ai/docs/framework/explanation-federated-evaluation.html
## Differential Privacy
https://flower.ai/docs/framework/explanation-differential-privacy.html
## Secure Aggregation Protocols
https://flower.ai/docs/framework/explanation-ref-secure-aggregation-protocols.html
## Flower Architecture
https://flower.ai/docs/framework/explanation-flower-architecture.html
## Flower Strategyy Abstraction
https://flower.ai/docs/framework/explanation-flower-strategy-abstraction.html
```python
```

```python
```

```python
```

```python
```