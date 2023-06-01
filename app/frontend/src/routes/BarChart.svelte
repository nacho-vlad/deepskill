
<script>

  export let white=0.32;
  export let black=0.5;
  export let draw=0.1;
  
  let yScale = 50;
  let topPadding = 5;

  let yTicks = [0, 0.25, 0.5, 0.75, 1]  
  
  function yTransform(tick) {
    return topPadding+yScale-tick*yScale;
  }
  
  
</script>

<div class={$$props.class}>
<svg viewBox="0 0 100 {topPadding+yScale+20}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xhtml="http://www.w3.org/1999/xhtml">
  <!-- Using g to inherit presentation attributes -->
  <g class="y-axis">
  
  {#each yTicks as tick}
    <g class="tick" transform="translate(0, {yTransform(tick)})">
      <line x2="100%"></line>
			<text y="-2">{tick}</text>
    </g>
  
  {/each}
  
  </g>

  <g fill="#f8e0b0" stroke="#f8e0b0" stroke-width="1">    
      <rect x="15" y="{topPadding+yScale-white*yScale}" width="10" height="{white*yScale}">
      <title>{white}</title>
      </rect>      
  </g>

  <g fill="#a37043" stroke="#a37043" stroke-width="1">    
      <rect x="45" y="{topPadding+yScale-black*yScale}" width="10" height="{black*yScale}">
      <title>{black}</title>
      </rect>
  </g>

  <g fill="gray" stroke="gray" stroke-width="1">    
      <rect x="75" y="{topPadding+yScale-draw*yScale}" width="10" height="{draw*yScale}">
      <title>{draw}</title>
      </rect>
  </g>



      <text x="10" y="{topPadding+yScale+10}" class="barLabel" lengthAdjust="spacingAndGlyphs" textLength="20">White</text>
      <text x="40" y="{topPadding+yScale+10}" class="barLabel" lengthAdjust="spacingAndGlyphs" textLength="20">Black</text>
      <text x="70" y="{topPadding+yScale+10}" class="barLabel" lengthAdjust="spacingAndGlyphs" textLength="20">Draw</text>      
</svg>
</div>

<style>
      .barLabel {
        font-size: 9px;
        line-height: 2px;
      }
      
      .tick {
    		font-size: .3em;
    		font-weight: 100;
    	}

    	.tick line {
    		stroke: #e2e2e2;
        stroke-width: 0.5;
    		stroke-dasharray: 2;
    	}

    	.tick text {
    		fill: #ccc;
    		text-anchor: start;
    	}

</style>
