from crewai import Agent, Task, Crew, Process
from crewai.project import agent, task, crew, CrewBase
from crewai_tools import FileReadTool, DirectoryReadTool
from pydantic import BaseModel
import json

class Result(BaseModel):
    expected_coverage: int
    feedback: str
    pass_fail: str  

from crewai import LLM

llm_openai_1 = LLM(model='gpt-4o-mini', temperature=0)
# For the static logic tester, use a more powerful model with reasoning capabilities
llm_reasoning = LLM(model='gpt-4o-mini', temperature=0)

"The Team"

@CrewBase
class EnhancedGenerator:
    """This crew is responsible for the jest case generation, mocking strategy, and static logic analysis"""

    @agent
    def directory_structure_agent(self) -> Agent:
        return Agent(
            role="Project Architecture Cartographer",
            goal="Create a comprehensive map of the project's structure and component relationships to facilitate effective Jest test implementation",
            backstory="I specialize in interpreting complex software architectures by analyzing directory structures, file relationships, and dependency patterns. With extensive experience mapping full-stack applications, I can identify the architectural patterns being used, distinguish between frontend and backend components, recognize Jest test frameworks, and understand how different parts of the application interconnect. My insights provide the foundation for effective code segmentation and Jest testing strategies.",
            llm=llm_openai_1,
            tools=[DirectoryReadTool('/content/gcc-national-registry-dashboard-Dev_Branch'), FileReadTool()]
        )
    
    @task
    def directory_structure_task(self) -> Task:
        return Task(
            description="""
            Analyze the project's directory structure to create a comprehensive architectural map with a focus on Jest testability:
            
            1. Generate a hierarchical representation of the directory structure
            2. Classify directories and files by their purpose (frontend, backend, config, tests, etc.)
            3. Identify the technology stack based on file extensions and configuration files
            4. Map dependencies between components by analyzing import statements and package configs
            5. Locate authentication-related code and understand its integration points
            6. Identify database models and data access layers
            7. Recognize API endpoints and service boundaries
            8. Detect existing Jest test files, test utilities, and mock implementations
            
            Pay special attention to architectural patterns that reveal component boundaries, such as:
            - React component structure for Jest/React Testing Library tests
            - Node.js modules suitable for Jest mocking
            - Potential areas for Jest snapshot testing
            - Service layers and dependency injection patterns that require Jest mocks
            """,
            expected_output="""
            A detailed architectural map containing:
            
            1. Directory Structure: A hierarchical representation of folders and key files
            2. Component Classification: Identification of frontend, backend, and shared components
            3. Technology Stack: List of frameworks, languages, and key libraries in use
            4. Dependency Graph: Visualization of how components interconnect
            5. Authentication Flow: Identification of auth-related components and their relationships
            6. Data Layer: Database schemas, models, and data access components
            7. API Surface: Endpoints, controllers, and service boundaries
            8. Testing Context: Existing Jest test structure, mock utilities, and test helpers
            9. Segmentation Recommendations: Natural boundaries for code segmentation based on Jest testing best practices
            
            The output should be structured in a way that the Code Segmentation Agent can use to make informed decisions about logical separation of components for Jest testing.
            """,
            agent=self.directory_structure_agent()
        )
    
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
            goal="Create comprehensive Jest mock objects and test fixtures that simulate real-world interactions for all external dependencies",
            backstory="I've specialized in creating realistic Jest test environments for complex applications. With deep knowledge of Jest's mocking capabilities including jest.mock(), jest.fn(), mockImplementation(), and spyOn(), I can simulate databases, authentication systems, APIs, and other external dependencies with precision. My expertise allows for testing components in isolation while maintaining realistic behavior of their dependencies. I'm particularly skilled at mocking security contexts and authentication flows in full-stack applications using Jest's powerful mocking framework.",
            llm=llm_openai_1
        )

    @task
    def mock_generator_task(self) -> Task:
        return Task(
            description="""
            For each code segment identified by the Code Structure Analyst, create appropriate Jest mocks and test fixtures:
            
            1. Identify all external dependencies (databases, APIs, services, etc.)
            2. Create Jest mock objects using appropriate Jest methods:
               - jest.mock() for module-level mocking
               - jest.fn() for function-level mocking
               - mockImplementation() for custom behavior
               - spyOn() for monitoring calls while preserving implementation
            3. Generate realistic test data that covers various scenarios including:
               - Authentication states (logged in, logged out, different permission levels)
               - Database responses (found records, empty results, errors)
               - API responses (success, failure, timeout, malformed responses)
               - User input variations (valid, invalid, edge cases)
            4. Ensure Jest mocks maintain the contract expected by the code segment
            5. Create Jest mock scenarios for happy paths and error conditions
            6. For authentication components, create Jest test fixtures representing different user roles and permissions
            
            Focus on creating Jest mocks that are:
            - Compatible with Jest's mocking system (properly using jest.mock syntax)
            - Realistic enough to test real behaviors
            - Controllable to simulate specific conditions using mockImplementation or mockReturnValue
            - Verifiable to confirm interactions occurred correctly with expect().toHaveBeenCalled assertions
            """,
            expected_output="""
            For each code segment, provide:
            
            1. Jest Mock Objects:
               - Complete Jest mock definition for each external dependency
               - Jest mock configuration settings using proper Jest syntax
               - Jest verification points to assert correct interaction
               
            2. Jest Test Fixtures:
               - Sample data for Jest tests representing different scenarios
               - User contexts for Jest authentication testing
               - Environment configurations for different Jest test conditions
               
            3. Jest Scenario Matrix:
               - Mapping of which Jest mocks and fixtures apply to which test scenarios
               - Expected outcomes for each Jest test scenario
               
            All mocks should use proper Jest syntax (jest.mock, jest.fn, etc.) and follow Jest best practices for mocking.
            The output should be structured to directly feed into the Jest Test Case Generator's process.
            """,
            agent=self.mock_generator_agent(),
            context=[self.directory_structure_task(),self.code_segmentation_task()]
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
                self.directory_structure_agent(),
                self.code_segmentation_agent(),
                self.mock_generator_agent(),
                self.test_case_generator_agent(),
                self.static_logic_tester_agent()
            ],
            tasks=[
                self.directory_structure_task(),
                self.code_segmentation_task(),
                self.mock_generator_task(),
                self.test_case_generator_task(),
                self.static_logic_analysis_task()
            ],
            process=Process.sequential,
            verbose=True
        )
