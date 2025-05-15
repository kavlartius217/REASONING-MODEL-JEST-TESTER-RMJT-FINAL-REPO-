from crewai import Agent, Task, Crew, Process
from crewai.project import agent, task, crew, CrewBase
from crewai_tools import FileReadTool, DirectoryReadTool
from pydantic import BaseModel
import json

class Result(BaseModel):
    expected_coverage: int
    feedback: str
    pass_fail: str  


#mocking tool
from langchain_community.graphs import Neo4jGraph
graph=Neo4jGraph()
from langchain.chains import GraphCypherQAChain

from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model='gpt-4o-mini')

chain=GraphCypherQAChain.from_llm(graph=graph,llm=llm,allow_dangerous_requests=True,verbose=True)


from langchain.tools import Tool
mocking_tool = Tool(
    name="mocking_tool",
    description=(
        "Use this tool to query the Neo4j knowledge graph using natural language to obtain insights about "
        "interdependent code files, functions, modules, and their relationships. "
        "It is designed to assist in generating accurate Jest mocks based on code dependencies, APIs, and usage flows. "
        "This tool simplifies the mock generation process by interpreting user queries in natural language and "
        "returning structured mock data, which can then be utilized by other agents to create unit test cases. "
        "Ideal for automating Jest mock creation for isolated unit testing in complex codebases."
    ),
    func=chain.invoke
)

#crewai conversion
from crewai.tools import BaseTool
from pydantic import Field

class MockingTool(BaseTool):
    name: str = "Mocking_Information_Tool"
    description: str = (
        "Use this tool to get information on code files that import from one another. "
        "You can ask queries in natural language, and the tool will query the Neo4j knowledge graph "
        "to return structured details about code dependencies, modules, and functions. "
        "This helps generate Jest mocks for isolated unit testing."
    )
    mocking: any = Field(default_factory=lambda: mocking_tool)

    def _run(self, query: str) -> str:
        """Execute the natural language query and return the structured mocking information."""
        try:
            return self.mocking.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"
#mocking tool


#crew starts
from crewai import LLM

llm_openai_1 = LLM(model='gpt-4o-mini', temperature=0)
# For the static logic tester, use a more powerful model with reasoning capabilities
llm_reasoning = LLM(model='gpt-4o-mini', temperature=0)

"The Team"

@CrewBase
class EnhancedGenerator:
    """This crew is responsible for the jest case generation, mocking strategy, and static logic analysis"""
          
    @agent
    def code_segmentation_agent(self) -> Agent:
        return Agent(
            role="Code Structure Analyst for Jest Testing",
            goal="Break down source code into logical, isolated segments that can be independently tested with Jest",
            backstory="As a code architect specializing in software decomposition for Jest testing, I analyze complex codebases and identify logical boundaries. With years of experience in various programming paradigms, I can recognize patterns, understand dependencies, and isolate functional units for effective Jest unit and integration tests. My expertise in full-stack applications helps me identify the natural divisions between components in both frontend and backend systems that align with Jest testing methodologies.",
            llm=llm_openai_1,
            tools=[FileReadTool('/content/gcc-national-registry-dashboard-Dev_Branch/server/src/controller/auth.js')]
        )
    
    @task
    def code_segmentation_task(self) -> Task:
        return Task(
            description="""
            Analyze the provided source code and segment it into logical, Jest-testable units. 
            
            IMPORTANT: For each segment, you MUST include:
            1. The actual code snippet of the segment (copy the exact code)
            2. Its primary function/purpose
            3. Its inputs and outputs
            4. Its dependencies that will need Jest mocking
            5. Potential edge cases or areas of concern
            
            Pay special attention to authentication flows, database interactions, and API endpoints. Each segment should be isolated enough to test independently with appropriate Jest mocks and test utilities.
            
            YOUR OUTPUT MUST INCLUDE THE ACTUAL CODE FOR EACH SEGMENT. This is critical for subsequent tasks.
            """,
            expected_output="""
            A structured list of code segments optimized for Jest testing. Each segment MUST contain:
            
            ```
            ## Segment Name: [name]
            
            ### Location
            [file path and line numbers]
            
            ### Code
            ```javascript
            // PASTE THE ACTUAL CODE HERE
            ```
            
            ### Functional Description
            [description of what this segment does]
            
            ### Inputs and Outputs
            - Inputs: [list inputs]
            - Outputs: [list outputs]
            
            ### Dependencies for Mocking
            [list dependencies that need Jest mocking]
            
            ### Testing Focus Areas
            [recommendations for Jest testing approaches]
            ```
            
            ENSURE THAT EACH SEGMENT INCLUDES THE ACTUAL CODE. The code must be presented exactly as it appears in the source file.
            """,
            create_directory=True,
            output_file="rmjt_tests/code.js",
            agent=self.code_segmentation_agent()
        )

    @agent
    def mock_generator_agent(self) -> Agent:
        return Agent(
            role="Jest Mock Specialist",
            goal="Create comprehensive Jest mock objects and test fixtures that simulate real-world interactions by analyzing both knowledge graph metadata and actual code files",
            backstory="I've specialized in creating realistic Jest test environments for complex applications. With deep knowledge of Jest's mocking capabilities including jest.mock(), jest.fn(), mockImplementation(), and spyOn(), I can simulate databases, authentication systems, APIs, and other external dependencies with precision. I combine knowledge graph metadata with direct code analysis to ensure my mocks accurately reflect actual component implementations and interactions. My expertise allows for testing components in isolation while maintaining realistic behavior of their dependencies. I'm particularly skilled at mocking security contexts and authentication flows in full-stack applications using Jest's powerful mocking framework.",
            llm=llm_openai_1,
            tools=[MockingTool(), FileReadTool()]  # Add both tools to the agent
        )

    @task
    def mock_generator_task(self) -> Task:
        return Task(
            description="""
            For each code segment identified by the Code Structure Analyst, create appropriate Jest mocks and test fixtures:
            
            1. Use the Mocking_Information_Tool to query the knowledge graph for:
               - All imports and dependencies of the target component
               - Complexity and mockability metrics
               - Error handling patterns that need to be tested
               - Relationships between the component and other parts of the system
               - File paths of the target component and its dependencies
            
            2. For each file path identified in the knowledge graph:
               - Use the FileReadTool to retrieve the actual code content using the file path
               - Read and analyze the source code to understand implementation details
               - Identify function signatures, parameter types, and return values
               - Examine actual implementation details to ensure mocks maintain the same interface
               - Note any specific error handling or edge cases in the implementation
            
            WORKFLOW PATTERN:
            - First use Mocking_Information_Tool to get metadata and relationships
            - Then use FileReadTool with the file paths to read the actual source code
            - Combine both sources of information for comprehensive mocking
            
            3. Identify all external dependencies (databases, APIs, services, etc.) using both the knowledge graph data and code analysis
            
            4. Create Jest mock objects using appropriate Jest methods based on dependency relationships and code examination:
               - jest.mock() for module-level mocking
               - jest.fn() for function-level mocking
               - mockImplementation() for custom behavior based on actual implementation
               - spyOn() for monitoring calls while preserving implementation
            
            5. Generate realistic test data that covers various scenarios including:
               - Authentication states (logged in, logged out, different permission levels)
               - Database responses (found records, empty results, errors)
               - API responses (success, failure, timeout, malformed responses)
               - User input variations (valid, invalid, edge cases)
            
            6. Ensure Jest mocks maintain the exact contract expected by the code segment based on code analysis
            
            7. Create Jest mock scenarios for happy paths and error conditions, particularly focusing on error handling patterns found in the code
            
            8. For authentication components, create Jest test fixtures representing different user roles and permissions
            
            9. Analyze component complexity from both the knowledge graph and code analysis to determine the appropriate level of mocking detail
            
            10. For React components, pay special attention to props, state management, hooks, and context usage to ensure mocks reflect actual component behavior
            
            Focus on creating Jest mocks that are:
            - Compatible with Jest's mocking system (properly using jest.mock syntax)
            - Realistic enough to test real behaviors
            - Faithful to the actual implementation details revealed in the code files
            - Controllable to simulate specific conditions using mockImplementation or mockReturnValue
            - Verifiable to confirm interactions occurred correctly with expect().toHaveBeenCalled assertions
            """,
            expected_output="""
            For each code segment, provide:
            
            1. Code Analysis:
               - Summary of code examination findings using the file paths from the knowledge graph
               - Key interfaces and function signatures that must be preserved in mocks
               - Implementation details relevant to mocking strategy
            
            2. Knowledge Graph Analysis:
               - Summary of component dependencies discovered through the knowledge graph
               - Mockability assessment based on graph data
               - Identified import relationships requiring mocking
            
            3. Jest Mock Objects:
               - Complete Jest mock definition for each external dependency
               - Jest mock configuration settings using proper Jest syntax
               - Jest verification points to assert correct interaction
               - Implementation details based on actual code analysis
               
            4. Jest Test Fixtures:
               - Sample data for Jest tests representing different scenarios
               - User contexts for Jest authentication testing
               - Environment configurations for different Jest test conditions
               - Data structures that match those used in the actual implementation
               
            5. Jest Scenario Matrix:
               - Mapping of which Jest mocks and fixtures apply to which test scenarios
               - Expected outcomes for each Jest test scenario
               - Coverage of error handling patterns identified in the code
               
            All mocks should use proper Jest syntax (jest.mock, jest.fn, etc.) and follow Jest best practices for mocking.
            The output should be structured to directly feed into the Jest Test Case Generator's process.
            """,
            agent=self.mock_generator_agent(),
            context=[self.code_segmentation_task()]
        )
    
    @agent
    def test_case_generator_agent(self) -> Agent:
        return Agent(
            role="Jest Test Architect",
            goal="Create comprehensive Jest test cases that verify functionality, edge cases, and error handling for each code segment",
            backstory="I'm an expert in Jest testing, with extensive experience in test-driven development and behavior-driven design using the Jest framework. I craft Jest tests that not only verify functionality but also document the expected behavior of systems. I specialize in full-stack application testing with Jest, React Testing Library, and Supertest, with particular attention to user flows, data validation, and security considerations. My Jest test cases balance thoroughness with practicality, ensuring critical paths are well-tested without creating excessive maintenance burden.",
            llm=llm_openai_1
        )

    @task
    def test_case_generator_task(self) -> Task:
        return Task(
            description="""
            Using the code segments from the Code Structure Analyst and the mocks from the Jest Mock Specialist, create comprehensive Jest test cases:
            
            1. For each code segment, develop a suite of Jest tests covering:
               - Basic functionality (happy path) using standard Jest assertions
               - Input validation and boundary conditions with appropriate Jest matchers
               - Error handling and edge cases using Jest's exception testing
               - Security considerations specific to Jest testing
               - Performance concerns where relevant using Jest's timer mocks
               
            2. Each Jest test case should specify:
               - Jest describe/it structure with clear test descriptions
               - Jest beforeEach/afterEach setup and teardown procedures
               - Test inputs and parameters
               - Which Jest mocks and fixtures to use
               - Expected outcomes with specific Jest assertions (expect().toBe, etc.)
               - Jest cleanup actions if needed
               
            3. For authentication-related functionality:
               - Jest test cases for login, logout, session management
               - Permission verification and access control tests
               - Token handling and security measures tests using Jest mocks
               
            4. For data handling components:
               - Jest data validation test cases
               - CRUD operation verification with Jest mocks
               - Data transformation tests with appropriate Jest assertions
               
            5. For user interfaces:
               - React Testing Library test cases for component rendering
               - Event handling tests using fireEvent or userEvent
               - UI state management tests with appropriate queries
               
            6. When {feedback} is received:
               - Parse the feedback variable for specific requested changes
               - Make those exact changes to the test code as requested
               - Update assertions, test structure, or mocks according to feedback
               - Document which feedback items were addressed and how they were implemented
               - Re-evaluate test coverage after implementing feedback changes
               - Ensure all feedback-driven changes maintain Jest best practices
            """,
            expected_output="""
            A structured set of Jest test cases for each code segment, including:
            
            1. Jest Test Suite Structure:
               - Logical grouping of tests using Jest's describe blocks
               - Setup and teardown procedures using beforeEach/afterEach
               
            2. Individual Jest Test Cases:
               - Test name and description in it/test blocks
               - Preconditions and Jest environment setup
               - Test input data and parameters
               - Jest mock configuration and behavior
               - Step-by-step execution process
               - Expected outcomes with specific Jest assertions
               - Edge cases and variations covered by separate tests
               
            3. Jest Coverage Analysis:
               - Assessment of test coverage for each code segment
               - Identification of untested or under-tested paths
               - Recommendations for Jest coverage settings
               
            4. Jest Testing Recommendations:
               - Prioritized list of Jest test cases by importance
               - Suggestions for additional Jest tests that may be valuable
               - Jest configuration options that might be beneficial
               
            5. Feedback Implementation Report (if feedback was provided):
               - List of feedback items that were addressed
               - Description of changes made to implement each feedback item
               - Explanation of how feedback improved test quality or coverage
               - Any feedback items that couldn't be implemented and why
               
            All test code should use proper Jest syntax and follow Jest best practices.
            If feedback was provided, the final code should reflect all requested changes.
            """,
            agent=self.test_case_generator_agent(),
            create_directory=True,
            output_file="rmjt_tests/code.test.js",
            context=[self.code_segmentation_task(),
                self.mock_generator_task()]
        )

    @agent
    def static_logic_tester_agent(self) -> Agent:
        return Agent(
            role="Jest Static Logic Analyzer",
            goal="Analyze a single Jest test file and its source code to identify logical issues without execution",
            backstory="""I am a deep reasoning expert specialized in static analysis of Jest test suites. With extensive knowledge of JavaScript, Jest's mocking system, and software testing principles, I can identify logical flaws in test cases by carefully analyzing the code flow, mock implementations, and test assertions without needing to run the tests.""",
            llm=llm_reasoning,
            tools=[FileReadTool('/content/rmjt_tests/code.js'),FileReadTool('/content/rmjt_tests/code.test.js')]
        )

    @task
    def static_logic_analysis_task(self) -> Task:
        return Task(
            description="""
            Perform a comprehensive static analysis of the single Jest test file and its relationship to the source code.
            
            The test file and source code have been generated by a previous crew and are available in the working directory.
            
            IMPORTANT: Assume that all mocks in the test are correctly implemented. Focus on analyzing the test logic itself.
            
            Without executing the test, analyze:
            
            1. Each test case's logical flow from setup through execution to assertions
            2. Verify test configurations properly use mocks and assertions match expected outcomes
            3. Identify contradictions or impossible conditions in the test logic
            4. Find potential coverage gaps and untested edge cases
            5. For each issue found:
               - Explain the logical problem step by step
               - Show why it would fail
               - Provide specific code changes to fix the issue
            
            YOUR OUTPUT MUST BE STRUCTURED AS A PYDANTIC MODEL WITH THE FOLLOWING FIELDS:
            - expected_coverage: An integer percentage (0-100) indicating how much of the source code functionality is covered by the tests
            - feedback: A detailed string containing all identified issues and SPECIFIC CODE CHANGES to fix each issue
            - pass_fail: Either "PASS" if the tests would execute successfully or "FAIL" if issues were found
            
            The feedback field should include actual code snippets showing both the problematic code and the corrected version.
            """,
            expected_output="""
            A JSON object strictly conforming to the Result pydantic model:
            
            {
                "expected_coverage": 75,  # Example percentage between 0-100
                "feedback": "Issue 1: [Description of issue]\\n\\nCurrent code:\\n```javascript\\n// Problematic code\\n```\\n\\nRecommended fix:\\n```javascript\\n// Fixed code\\n```\\n\\nIssue 2: [Description]...",
                "pass_fail": "FAIL"  # Either "PASS" or "FAIL"
            }
            
            The feedback must include SPECIFIC CODE CHANGES for each issue, showing both the original problematic code and your recommended fixed code.
            """,
            agent=self.static_logic_tester_agent(),
            output_pydantic=Result
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.code_segmentation_agent(),
                self.mock_generator_agent(),
                self.test_case_generator_agent(),
                self.static_logic_tester_agent()
            ],
            tasks=[
                self.code_segmentation_task(),
                self.mock_generator_task(),
                self.test_case_generator_task(),
                self.static_logic_analysis_task()
            ],
            process=Process.sequential,
            verbose=True
        )
