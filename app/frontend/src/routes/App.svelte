<svelte:head>
  <link href="https://fonts.googleapis.com/css?family=EB+Garamond" rel="stylesheet">
</svelte:head>


<script>
  import 'chota';
  import BarChart from './BarChart.svelte';
  
  const url = "http://127.0.0.1:8000";
  
  let prediction = {
    "white": 0,
    "black": 0,
    "draw": 0,
  }
  
  let white = "";
  let black = "";
  let minutes = null;
  let increment = null;
  
  let error = {
    'white': '',
    'black': '',
    'min': '',
    'inc': '',
  }
  
  async function fetchPrediction() {
        
    const response = await fetch(url + `/predict?white=${white}&black=${black}&min=${minutes}&inc=${increment}`);
    if (!response.ok) {        
      const err = await response.json();
      console.log(`Server returned an error: ${err.detail}`);
      const invalid = err.detail
      for (const element of invalid) {
        error[element] = 'error'
      }
      error = error
      
      return;
    }
    
    prediction = await response.json();
  }
  
  function resetInput(input) {
    error[input] = ''
    error = error
  }
  
</script>

<div style="width:47.5%; margin:auto">

  <h1>
    DeepSkill
  </h1>
  <h2>
    Rating Players with Temporal Graph Networks
  </h2>
 
  <div class="time-control">
    <label for="Min"> Time Control </label>
    <div class="grid-container">
      <input bind:value={minutes} class={error['min']} on:input={()=>resetInput('min')} min=1 type="number" placeholder="Mins" id="Min">
      <input bind:value={increment} class={error['inc']} on:input={()=>resetInput('inc')} min=0 max=60 type="number" placeholder="Inc" id="Inc">
    </div>
  </div>
  
  <div class="grid-container" style="margin-bottom:20px">
  
    <div class="grid-item">
    <label for="White">White</label>
    <input bind:value={white} class={error['white']} on:input={()=>resetInput('white')} placeholder="Lichess Username" id="White">
    </div>
  
    <div class="grid-item">
    <label style="text-align: center;" for="Black">Black</label>
    <input bind:value={black} class={error['black']} on:input={()=>resetInput('black')} placeholder="Lichess Username" id="Black">
    </div>

  </div>
  
  <button on:click={fetchPrediction} style="width: 30%; margin:auto; display:block">Predict</button>
  
  <h4>
    Outcome Probabilities
  </h4>
  <BarChart class="chart" white={prediction["white"]} black={prediction["black"]} draw={prediction["draw"]} />
  
  <h2 style="margin-bottom:10px">
    About
  </h2>
  <p>
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
  </p>
  
</div>


<style>

  .grid-container {
    display: grid;
    grid-template-columns: auto auto;
    column-gap: 20px;
  }
  
  .time-control {
    width: 35%;
    margin: auto;
  }
  
  :global(.chart) {
    width: 50%;
    margin: auto;
  }
  
  label {
    display: block;
    text-align: center;
  }
  
  * {
    font-size: 1.1em;
    font-family: "EB Garamond", sans-serif
  }
  
  h1 {
    font-size:5em; 
    margin:0px; 
    margin-top: 20px; 
    padding: 0px; 
    line-height:120%;
    font-family: "EB Garamond", sans-serif
  }
  
  h2 {
    font-size:3em; 
    width: 80%; 
    margin:0px; 
    margin-bottom: 40px;
    padding:0px; 
    line-height:120%;
    font-family: "EB Garamond", sans-serif
  }
  
  h4 {
    font-size:1.8em;
    width: 50%;
    margin: auto;
    text-align:center;
    margin-top: 20px;
  }
  
  .error {
    border-color: red,
  }

</style>


