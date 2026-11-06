"""
Export human baseline questions to Qualtrics survey.

Creates a survey with 5 blocks (conditions):
- Block 1: Control (no manipulation)
- Block 2: Peer pressure with "A few"
- Block 3: Peer pressure with "Almost all"
- Block 4: Negative incentive with "10%"
- Block 5: Negative incentive with "90%"

Each question includes JavaScript to:
- Prevent text selection
- Prevent copy/paste (keyboard shortcuts and context menu)
- Enforce minimum 40 word response

Survey Flow (automatically configured):
- Randomizer shows 3 of 5 blocks to each participant (configurable)
- Which 3 blocks is randomized with "Evenly Present Elements"
- Order of the 3 blocks is also randomized
- With 500 questions, 5 responses each, 3 questions/participant = ~834 participants

MANUAL STEP REQUIRED:
- Within-block question randomization (1 random question per block) must be
  configured manually in Qualtrics UI - the API doesn't support this setting.
  Instructions are printed when the script runs.

Usage:
    # Dry run (validates data without creating survey)
    uv run data_generation/human_baseline/export_to_qualtrics.py path/to/data.jsonl --dry_run true
    
    # Create new survey
    uv run data_generation/human_baseline/export_to_qualtrics.py path/to/data.jsonl
    
    # Add blocks to existing survey
    uv run data_generation/human_baseline/export_to_qualtrics.py path/to/data.jsonl --survey_id SV_xxxxxxxxxxxxx
    
    # Custom number of blocks per participant (default is 3)
    uv run data_generation/human_baseline/export_to_qualtrics.py path/to/data.jsonl --blocks_per_participant 2
    
    # Disable randomizer setup (configure manually in Qualtrics)
    uv run data_generation/human_baseline/export_to_qualtrics.py path/to/data.jsonl --setup_randomizer false
"""

import os
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from jsonargparse import CLI

from fraudbench.data_generation import qualtrics as qq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths
DEFAULT_IDS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), '..', 'data', 'baseline_ids.csv')
DEFAULT_IDS_PATH = os.path.normpath(DEFAULT_IDS_PATH)

# Block configurations
BLOCK_CONFIGS = [
    {
        "name": "control",
        "description": "Block 1: Control Condition",
        "condition_field": None,
        "quantity": None,
    },
    {
        "name": "peer_few",
        "description": "Block 2: Peer Pressure (A few)",
        "condition_field": "experiment_peer_pressure",
        "quantity": "A few",
    },
    {
        "name": "peer_almost_all",
        "description": "Block 3: Peer Pressure (Almost all)",
        "condition_field": "experiment_peer_pressure",
        "quantity": "Almost all",
    },
    {
        "name": "incentive_10pct",
        "description": "Block 4: Negative Incentive (10%)",
        "condition_field": "experiment_negative_incentive",
        "quantity": "10%",
    },
    {
        "name": "incentive_90pct",
        "description": "Block 5: Negative Incentive (90%)",
        "condition_field": "experiment_negative_incentive",
        "quantity": "90%",
    },
]

# Minimum word count for responses
MIN_WORD_COUNT = 40


def get_anti_copy_paste_js() -> str:
    """
    Generate JavaScript to prevent copy/paste and text selection,
    display a live word count, and enforce minimum word count.
    
    Based on: https://gist.github.com/marketinview/aaf237fea91d2f7947fa4f170ac16486
    
    Returns:
        JavaScript code string to attach to questions.
    """
    js_code = qq.QuestionJS()
    
    # On page load: set up word count display and hide next button
    # Using jQuery approach from the Gist
    js_code.on_load(f'''
        var qobj = this;
        var minWords = {MIN_WORD_COUNT};
        var q = jQuery("#" + this.questionId);
        var input = q.find(".InputText");
        
        // Add word count display (adapted from Gist)
        input.after("<div style='font-size:0.9em;margin-top:8px;color:#666'>Word count: <span class='wordCount'>0</span> / " + minWords + " minimum required</div>");
        var display = q.find(".wordCount");
        
        // Initially hide the next button
        qobj.hideNextButton();
        
        // Word count function (adapted from Gist)
        function countWords(el) {{
            var wordCount = 0;
            if (el.value.trim().length > 0) {{
                var matches = el.value.trim().match(/\\S+/g);
                wordCount = matches ? matches.length : 0;
            }}
            display.text(wordCount);
            
            // Update styling and button visibility based on word count
            if (wordCount >= minWords) {{
                display.css({{'color': 'green', 'font-weight': 'bold'}});
                display.parent().css('color', 'green');
                qobj.showNextButton();
            }} else {{
                display.css({{'color': '#666', 'font-weight': 'normal'}});
                display.parent().css('color', '#666');
                qobj.hideNextButton();
            }}
        }}
        
        // Initial count
        if (input.length > 0) {{
            countWords(input.get(0));
        }}
        
        // Update on input
        input.on("input", function() {{ countWords(this); }});
        
        // Trim whitespace on blur
        input.blur(function() {{ this.value = this.value.trim(); }});
        
        // Disable text selection on question container (but not input)
        q.css({{
            'user-select': 'none',
            '-webkit-user-select': 'none',
            '-moz-user-select': 'none',
            '-ms-user-select': 'none'
        }});
        
        // Allow selection in input
        input.css({{
            'user-select': 'text',
            '-webkit-user-select': 'text',
            '-moz-user-select': 'text',
            '-ms-user-select': 'text'
        }});
        
        // Disable paste
        input.on('paste', function(e) {{
            e.preventDefault();
            alert('Pasting is not allowed. Please type your response.');
            return false;
        }});
        
        // Disable copy and cut
        input.on('copy cut', function(e) {{
            e.preventDefault();
            return false;
        }});
        
        // Disable context menu (right-click)
        q.on('contextmenu', function(e) {{
            e.preventDefault();
            return false;
        }});
        
        // Disable keyboard shortcuts for copy/paste
        q.on('keydown', function(e) {{
            // Ctrl+C, Ctrl+V, Ctrl+X (Windows/Linux) or Cmd+C, Cmd+V, Cmd+X (Mac)
            if ((e.ctrlKey || e.metaKey) && (e.key === 'c' || e.key === 'v' || e.key === 'x' ||
                                              e.key === 'C' || e.key === 'V' || e.key === 'X')) {{
                e.preventDefault();
                if (e.key === 'v' || e.key === 'V') {{
                    alert('Pasting is not allowed. Please type your response.');
                }}
                return false;
            }}
        }});
    ''')
    
    return js_code.script()


def convert_quotes_to_html_entities(text: str) -> str:
    """Convert straight double quotes to ``&ldquo;``/``&rdquo;`` HTML entities.

    Qualtrics renders straight quotes inconsistently in rich-text questions,
    so we alternate opening/closing curly-quote entities by simple toggling
    (no support for nested or stray quotes).

    Args:
        text: Source text (may contain ``"`` characters).

    Returns:
        Text with each ``"`` replaced by ``&ldquo;`` or ``&rdquo;``,
        alternating starting from opening.
    """
    import re
    
    result = []
    in_quote = False
    
    for i, char in enumerate(text):
        if char == '"':
            if not in_quote:
                result.append('&ldquo;')
                in_quote = True
            else:
                result.append('&rdquo;')
                in_quote = False
        else:
            result.append(char)
    
    return ''.join(result)


def build_question_text(
    row: pd.Series,
    condition_field: Optional[str] = None,
    quantity: Optional[str] = None,
) -> str:
    """
    Build the full question text for a given row and condition.
    
    Args:
        row: DataFrame row containing the question data.
        condition_field: Field name for the condition ('experiment_peer_pressure' or 'experiment_negative_incentive').
        quantity: The value to replace {QUANTITY} placeholder with.
    
    Returns:
        Full question text as HTML string.
    """
    # Base components
    system_prompt = row['human_baseline_system_prompt']
    respondent_context = row['human_baseline_human_instructions']
    vignette = row['human_baseline_vignette']
    qa_open = row['human_baseline_query_open']
    
    # Build vignette text - append condition in same paragraph if applicable
    vignette_text = vignette
    if condition_field and quantity:
        condition_text = row[condition_field].replace("{QUANTITY}", quantity)
        vignette_text = f"{vignette} {condition_text}"
    
    # Convert double quotes to HTML entities for proper rendering in Qualtrics
    system_prompt = convert_quotes_to_html_entities(system_prompt)
    respondent_context = convert_quotes_to_html_entities(respondent_context)
    vignette_text = convert_quotes_to_html_entities(vignette_text)
    qa_open = convert_quotes_to_html_entities(qa_open)
    
    # Build question text with proper spacing
    # All plain text except the final question which is bold
    parts = [
        f"<p>{system_prompt} {respondent_context}</p>",
        "<br>",  # Extra line break between context and vignette
        f"<p>{vignette_text}</p>",
        "<br>",  # Extra line break before the question
        f"<p><strong>{qa_open}</strong></p>",  # Only the question is bold
    ]
    
    return "\n".join(parts)


def populate_existing_survey(
    api: qq.QualtricsSurveyDefinitionAPI,
    survey_id: str,
    survey: qq.BlockSurvey,
    setup_randomizer: bool = True,
    n_blocks_per_participant: int = 3,
) -> None:
    """
    Add blocks and questions from a BlockSurvey to an existing Qualtrics survey.
    
    Args:
        api: Qualtrics API connection.
        survey_id: ID of the existing survey.
        survey: BlockSurvey object containing blocks and questions to add.
        setup_randomizer: If True, configure survey flow to randomize blocks.
        n_blocks_per_participant: Number of blocks each participant sees.
    """
    import tqdm
    from fraudbench.data_generation.qualtrics.questions import PageBreak
    
    n_blocks = len(survey.blocks)
    # Count actual questions (not page breaks)
    n_items = sum(len(b.questions) for b in survey.blocks)
    n_questions = sum(1 for b in survey.blocks for q in b.questions if not isinstance(q, PageBreak))
    print(f"Adding to survey: {n_blocks} blocks, {n_questions} questions ({n_items} API calls)")
    sys.stdout.flush()
    
    # Track created block IDs for flow setup
    created_block_ids = []
    
    progress = tqdm.tqdm(total=n_blocks + n_items, dynamic_ncols=True, miniters=1)
    for block in survey.blocks:
        result = api.create_block(survey_id=survey_id, block_description=block.description)
        block_id = result['BlockID']
        created_block_ids.append((block_id, block.description))
        progress.update()
        for question in block.questions:
            question.create(api, survey_id, block_id=block_id)
            progress.update()
    progress.close()
    print(f"Survey {survey_id} populated with new blocks")
    
    # Set up randomizer in survey flow
    if setup_randomizer and created_block_ids:
        print(f"\nSetting up randomizer: each participant sees {n_blocks_per_participant} of {n_blocks} blocks...")
        setup_survey_flow_randomizer(api, survey_id, created_block_ids, n_blocks_per_participant)
        # Print instructions for within-block randomization (API doesn't support this)
        print_block_randomization_instructions(created_block_ids, questions_per_block=1)


def setup_survey_flow_randomizer(
    api: qq.QualtricsSurveyDefinitionAPI,
    survey_id: str,
    block_ids: list,
    n_samples: int,
) -> None:
    """
    Update survey flow to use a randomizer that shows n_samples blocks.
    
    Args:
        api: Qualtrics API connection.
        survey_id: Survey ID.
        block_ids: List of (block_id, description) tuples.
        n_samples: Number of blocks each participant should see.
    """
    # Get current flow
    current_flow = api.get_flow(survey_id)
    
    # Build randomizer flow structure
    # The flow structure has a root with Flow array containing elements
    randomizer_flow_id = "FL_random"
    
    # Create block flow elements for each block
    block_flows = []
    for i, (block_id, description) in enumerate(block_ids):
        block_flows.append({
            "Type": "Block",
            "ID": block_id,
            "FlowID": f"FL_{i+2}",  # Start from FL_2 since FL_1 might be used
            "Autofill": []
        })
    
    # Create the randomizer element
    randomizer_element = {
        "Type": "BlockRandomizer",
        "FlowID": randomizer_flow_id,
        "SubSet": n_samples,
        "EvenPresentation": True,
        "Flow": block_flows
    }
    
    # Build new flow with just the randomizer
    # Keep any existing elements like EndSurvey
    new_flow = {
        "FlowID": "FL_1",
        "Type": "Root",
        "Flow": [randomizer_element]
    }
    
    # Update the flow
    api.update_flow(survey_id, new_flow)
    print(f"Survey flow updated: randomizer will show {n_samples} of {len(block_ids)} blocks evenly")


def print_block_randomization_instructions(block_ids: list, questions_per_block: int = 1) -> None:
    """
    Print instructions for manually setting up within-block question randomization.
    
    Note: The Qualtrics API does not support setting block-level question randomization
    via the update_block endpoint. This must be configured manually in the Qualtrics UI.
    
    Args:
        block_ids: List of (block_id, description) tuples.
        questions_per_block: Number of questions each participant sees per block.
    """
    print(f"\n" + "="*70)
    print("MANUAL STEP REQUIRED: Within-Block Question Randomization")
    print("="*70)
    print(f"\nThe Qualtrics API doesn't support setting block-level randomization.")
    print(f"Please configure each block to show {questions_per_block} random question(s):\n")
    print("For EACH of the 5 blocks:")
    print("  1. Click on the block name to open Block Options")
    print("  2. Select 'Question Randomization'")
    print(f"  3. Choose 'Present only {questions_per_block} of total questions'")
    print("  4. Check 'Evenly Present Elements'")
    print("  5. Click 'Save'\n")
    print("Blocks to configure:")
    for block_id, description in block_ids:
        print(f"  - {description} ({block_id})")
    print("="*70 + "\n")


def create_block(
    df: pd.DataFrame,
    block_config: dict,
    js_code: str,
) -> qq.Block:
    """
    Create a Qualtrics block with questions for the given condition.
    
    Args:
        df: DataFrame with filtered question data.
        block_config: Configuration dict for this block.
        js_code: JavaScript code to attach to each question.
    
    Returns:
        Qualtrics Block object.
    """
    block = qq.Block(description=block_config["description"])
    
    for _, row in df.iterrows():
        summary_id = row['summary_id']
        block_name = block_config["name"]
        
        # Build question text
        question_text = build_question_text(
            row,
            condition_field=block_config["condition_field"],
            quantity=block_config["quantity"],
        )
        
        # Create text entry question (essay size) with minimum word count validation
        question = qq.TextEntryQuestion(
            data_export_tag=f"{summary_id}_{block_name}",
            text_html=question_text,
            script_js=js_code,
            force_response=False,
            size_of_response="essay",
            min_words=MIN_WORD_COUNT,
            min_words_message=f"Please enter at least {MIN_WORD_COUNT} words before continuing.",
        )
        
        block.append_question(question)
        block.append_page_break()
    
    return block


def export_to_qualtrics(
    data: str,
    ids: Optional[str] = None,
    survey_name: str = "FraudBench Human Baseline",
    survey_id: Optional[str] = None,
    dry_run: bool = False,
    setup_randomizer: bool = True,
    blocks_per_participant: int = 3,
) -> Optional[str]:
    """
    Export human baseline questions to Qualtrics.
    
    Args:
        data: Path to JSONL file containing human baseline data.
        ids: Path to CSV file with summary_ids to include. Defaults to data/baseline_ids.csv.
        survey_name: Name for the Qualtrics survey (used only when creating new survey).
        survey_id: Optional existing survey ID to add blocks/questions to. If not provided,
                   creates a new survey.
        dry_run: If True, don't actually create the survey (just validate).
        setup_randomizer: If True, configure survey flow so each participant sees a random
                          subset of blocks. Defaults to True.
        blocks_per_participant: Number of blocks (conditions) each participant sees.
                               Each block shows 1 random question. Defaults to 3.
    
    Returns:
        Survey ID if created successfully, None otherwise.
    """
    # Load environment variables
    load_dotenv()
    
    api_key = os.environ.get('QUALTRICS_API_KEY')
    data_center = os.environ.get('QUALTRICS_DATA_CENTER_ID')
    
    if not api_key or not data_center:
        print("Error: QUALTRICS_API_KEY and QUALTRICS_DATA_CENTER_ID must be set in environment.")
        print("Please add these to your .env file.")
        return None
    
    # Set default IDs path
    if ids is None:
        ids = DEFAULT_IDS_PATH
    
    # Load data
    print(f"Loading data from {data}")
    df = pd.read_json(data, lines=True)
    print(f"Loaded {len(df)} rows")
    
    # Load and filter by IDs
    print(f"Loading IDs from {ids}")
    ids_df = pd.read_csv(ids)
    # Handle the trailing comma in header by using the first column
    id_column = ids_df.columns[0].rstrip(',')
    if id_column not in ids_df.columns:
        # Column name has trailing comma, rename it
        ids_df.columns = [c.rstrip(',') for c in ids_df.columns]
    
    valid_ids = set(ids_df['summary_id'].astype(str).tolist())
    print(f"Found {len(valid_ids)} valid IDs")
    
    # Filter dataframe
    df['summary_id'] = df['summary_id'].astype(str)
    df_filtered = df[df['summary_id'].isin(valid_ids)]
    print(f"Filtered to {len(df_filtered)} rows")
    
    if len(df_filtered) == 0:
        print("Error: No matching rows found after filtering by IDs.")
        return None
    
    required_cols = [
        'summary_id',
        'human_baseline_system_prompt',
        'human_baseline_human_instructions',
        'human_baseline_vignette',
        'human_baseline_query_open',
        'experiment_peer_pressure',
        'experiment_negative_incentive',
    ]
    missing_cols = [c for c in required_cols if c not in df_filtered.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # Generate JavaScript code for anti-copy/paste and word count validation
    print("Generating JavaScript for question protection...")
    js_code = get_anti_copy_paste_js()
    
    # Create survey with blocks
    print(f"Creating survey: {survey_name}")
    survey = qq.BlockSurvey(name=survey_name)
    
    # Create each block
    for block_config in BLOCK_CONFIGS:
        print(f"  Creating {block_config['description']}...")
        block = create_block(df_filtered, block_config, js_code)
        survey.append_block(block)
        print(f"    Added {len(df_filtered)} questions")
    
    total_questions = len(df_filtered) * len(BLOCK_CONFIGS)
    print(f"Total questions across all blocks: {total_questions}")
    
    if dry_run:
        print("\n[DRY RUN] Survey not created. Remove --dry_run flag to create survey.")
        return None
    
    # Create API connection
    print("\nConnecting to Qualtrics API...")
    api = qq.QualtricsSurveyDefinitionAPI(api_key, data_center)
    
    if survey_id:
        # Add to existing survey
        print(f"Adding blocks to existing survey: {survey_id}")
        populate_existing_survey(
            api, survey_id, survey,
            setup_randomizer=setup_randomizer,
            n_blocks_per_participant=blocks_per_participant,
        )
        print(f"\nBlocks added successfully to survey {survey_id}!")
    else:
        # Create new survey
        print("Uploading survey to Qualtrics...")
        survey_id = survey.create(api)
        print(f"\nSurvey created successfully!")
        print(f"Survey ID: {survey_id}")
        
        # Set up randomizer for new survey
        if setup_randomizer:
            print(f"\nSetting up randomizer: each participant sees {blocks_per_participant} of {len(BLOCK_CONFIGS)} blocks...")
            # Get block IDs from the survey
            survey_def = api.get_survey(survey_id)
            block_ids = [
                (bid, bdata.get('Description', ''))
                for bid, bdata in survey_def.get('Blocks', {}).items()
                if bdata.get('Type') == 'Standard'
            ]
            if block_ids:
                setup_survey_flow_randomizer(api, survey_id, block_ids, blocks_per_participant)
                # Print instructions for within-block randomization (API doesn't support this)
                print_block_randomization_instructions(block_ids, questions_per_block=1)
    
    print(f"\nNext steps:")
    print("1. Open the survey in Qualtrics web editor")
    if setup_randomizer:
        print(f"2. Verify Survey Flow: BlockRandomizer shows {blocks_per_participant} of 5 blocks (auto-configured)")
        print("3. IMPORTANT: Configure each block to show 1 random question (see instructions above)")
    else:
        print("2. Configure randomization manually (Survey Flow > Randomizer)")
        print("3. Configure each block to show 1 random question")
    print("4. Set up quotas if you need exactly 5 responses per question")
    print("5. Preview and test the survey")
    
    return survey_id


if __name__ == "__main__":
    CLI(export_to_qualtrics)
